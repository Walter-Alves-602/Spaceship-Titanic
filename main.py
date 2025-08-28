import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import randint

data = r"https://raw.githubusercontent.com/Walter-Alves-602/Spaceship-Titanic/refs/heads/main/train.csv"
df = pd.read_csv(data)


class DataRegenerator(BaseEstimator, TransformerMixin):
    """Regenera e imputa colunas conforme regras do notebook:\n    - extrai Group de PassengerId\n    - separa Cabin em deck/num/side\n    - preenche HomePlanet e VIP pela moda do grupo (fallback moda global)\n    - preenche numéricos por mediana condicional em VIP\n    - heurística para CryoSleep baseada em gastos\n    - preenche Destination por deck para D/E/F/T\n    - preenche modos restantes e cria algumas features simples\n    Retorna um DataFrame transformado."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        # nenhum aprendizado necessário, apenas retorna self
        return self

    def transform(self, X):
        # espera um pandas DataFrame e retorna outro DataFrame
        df = X.copy()
        # garantir Group a partir de PassengerId
        if 'Group' not in df.columns and 'PassengerId' in df.columns:
            df['Group'] = df['PassengerId'].apply(lambda x: int(x.split('_')[0]) if pd.notna(x) else np.nan)

        # separar Cabin em deck/num/side quando existir
        if 'Cabin' in df.columns:
            df[['deck', 'num', 'side']] = df['Cabin'].str.split('/', expand=True)

        # função auxiliar: preencher coluna categórica pela moda do grupo com fallback para moda global
        def fill_by_group_mode(col):
            if col not in df.columns:
                return
            mask = df[col].isna()
            groups_with_na = df.loc[mask, 'Group'].dropna().unique()
            for g in groups_with_na:
                mode_vals = df.loc[df['Group'] == g, col].mode()
                if not mode_vals.empty:
                    df.loc[(df['Group'] == g) & (df[col].isna()), col] = mode_vals[0]

        # ==========================
        # 2. Carregamento dos dados
        # ==========================

        # ==========================
        # 3. Classe de transformação customizada
        # ==========================

            # fallback para moda global
            if df[col].isna().sum() > 0:
                global_mode = df[col].mode()
                if not global_mode.empty:
                    df[col] = df[col].fillna(global_mode[0])

        # preencher HomePlanet e VIP por moda do grupo
        fill_by_group_mode('HomePlanet')
        fill_by_group_mode('VIP')

        # preencher numericos por mediana dependendo de VIP (quando existir)
        num_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Age"]
        num_cols = [c for c in num_cols if c in df.columns]
        if 'VIP' in df.columns and df['VIP'].notna().any():
            vip_mask = df['VIP'] == True
            for col in num_cols:
                med_vip = df.loc[vip_mask, col].median() if vip_mask.any() else np.nan
                med_notvip = df.loc[~vip_mask, col].median() if (~vip_mask).any() else np.nan
                df.loc[df[col].isna() & vip_mask, col] = med_vip
                df.loc[df[col].isna() & (~vip_mask), col] = med_notvip
        else:
            for col in num_cols:
                df[col] = df[col].fillna(df[col].median())

        # heurística para CryoSleep: se gastou 0 em RoomService ou ShoppingMall -> provável CryoSleep
        if 'CryoSleep' in df.columns:
            df.loc[((df.get('RoomService', 0) == 0) | (df.get('ShoppingMall', 0) == 0)) & (df['CryoSleep'].isnull()), 'CryoSleep'] = True
            df.loc[((df.get('RoomService', 0) != 0) | (df.get('ShoppingMall', 0) != 0)) & (df['CryoSleep'].isnull()), 'CryoSleep'] = False

        # preencher Destination por deck quando plausível
        if 'Destination' in df.columns and 'deck' in df.columns:
            df.loc[df['deck'].isin(['D', 'E', 'F', 'T']) & df['Destination'].isnull(), 'Destination'] = 'TRAPPIST-1e'

        # preencher modos restantes para colunas categóricas importantes
        for col in ('deck', 'num', 'side', 'Cabin', 'Destination'):
            if col in df.columns:
                if df[col].isna().sum() > 0:
                    df[col] = df[col].fillna(df[col].mode()[0])

        # features simples: TotalSpend e GroupSize
        spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        if all(c in df.columns for c in spend_cols):
            df['TotalSpend'] = df['RoomService'].fillna(0) + df['FoodCourt'].fillna(0) + df['ShoppingMall'].fillna(0) + df['Spa'].fillna(0) + df['VRDeck'].fillna(0)
        if 'Group' in df.columns:
            df['GroupSize'] = df.groupby('Group')['Group'].transform('count')

        df['side'] = df['side'].map({'P': -1, 'S': 1})
        df['deck'] = df['deck'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8})
        
        for col in ('CryoSleep', 'VIP', 'Transported'):
            df[col] = df[col].map({True: 1, False: -1})

        # converter colunas do tipo 'object' para dtypes mais específicos quando possível
        # evita o FutureWarning: Downcasting object dtype arrays on .fillna is deprecated
        # e aplica a futura semântica de conversão de tipos de forma explícita
        try:
            df = df.infer_objects(copy=False)
        except Exception:
            # infer_objects é seguro, mas em caso de falha, retornamos o df original
            pass

        return df
    
# aplicar primeiro o regenerator para garantir colunas criadas (Group, TotalSpend, etc.)
reg_pipeline = Pipeline([('regenerator', DataRegenerator())])
df_reg = reg_pipeline.fit_transform(df)
# garantir que temos um DataFrame (sklearn typing às vezes mostra ndarray)
if not isinstance(df_reg, pd.DataFrame):
    try:
        df_reg = pd.DataFrame(df_reg, columns=df.columns)
    except Exception:
        df_reg = pd.DataFrame(df_reg)

# agora construir listas de nomes de colunas (ColumnTransformer espera listas de nomes, não DataFrames)
categorical_cols = [c for c in ['HomePlanet', 'Destination', 'Group'] if c in df_reg.columns]
quantitative_cols = [c for c in ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'num', 'TotalSpend', 'GroupSize', 'deck'] if c in df_reg.columns]
binary_cols = [c for c in ['CryoSleep', 'VIP', 'side'] if c in df_reg.columns]
dropable_cols = [c for c in ['PassengerId', 'Cabin', 'Name'] if c in df_reg.columns]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, quantitative_cols),
        ('cat', categorical_transformer, categorical_cols),
        ('drop', 'drop', dropable_cols)
    ],
    remainder='passthrough'
)


full_pipeline = Pipeline(steps=[
    ('regenerator', DataRegenerator()),
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])


# === Randomized search para hiperparâmetros do classificador (leva alguns minutos) ===
# usar o DataFrame original como X porque a pipeline já contém o regenerator
X = df
y = df['Transported']

param_dist = {
    'clf__n_estimators': [200,250,300,350,400,450,500],
    'clf__max_depth': [None, 6, 10, 20, 30],
    'clf__min_samples_split': [2,5,7,9,11,13,15],
    'clf__min_samples_leaf': [1,2,3,4,5],
    'clf__max_features': ['sqrt', 'log2', 0.2, 0.5, None],
    'clf__class_weight': [None, 'balanced']
}

cv = StratifiedKFold(n_splits=5, shuffle=True)
rs = RandomizedSearchCV(full_pipeline, param_dist, n_iter=40, scoring='roc_auc',
                        cv=cv, n_jobs=-1, verbose=2)

print('Iniciando RandomizedSearchCV (pode levar alguns minutos)...')
rs.fit(X, y)
print('Best AUC:', rs.best_score_)
print('Best params:', rs.best_params_)
