
# In[1]:


# pip install --upgrade jupyterlab


# In[2]:


pip install shap


# In[3]:


pip install phik


# In[4]:


pip install fitter


# In[5]:


pip install -U scikit-learn


# In[6]:


pip install seaborn


# In[7]:


pip install -U numba


# In[8]:


pip install --upgrade pip


# In[9]:


pip install matplotlib==3.5.2


# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import shap
import matplotlib.cm

from IPython.display import display
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

from fitter import Fitter, get_common_distributions, get_distributions

#vif
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor 

#preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

#models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#metrics
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from phik.report import plot_correlation_matrix
from phik import phik_matrix
from phik.report import plot_correlation_matrix
from phik import report


warnings.filterwarnings("ignore", category=FutureWarning)


# In[11]:


market_file = pd.read_csv('/datasets/market_file.csv', sep = ',', decimal='.')
market_money = pd.read_csv('/datasets/market_money.csv', sep = ',', decimal='.')
market_time = pd.read_csv('/datasets/market_time.csv',  sep = ',', decimal='.')
money = pd.read_csv('/datasets/money.csv',  sep = ';', decimal=',')


# In[12]:


market_file.head(5)


# In[13]:


market_money.head(5)


# In[14]:


market_time.head(5)


# In[15]:


money.head(5)



# In[16]:


market_file.info()


# In[17]:


market_file.isna().mean()


# In[18]:


market_file.duplicated().sum()



# In[19]:


market_file['Покупательская активность'].unique()


# In[20]:


market_file['Тип сервиса'].unique()


# In[21]:


market_file['Разрешить сообщать'].unique()


# In[22]:


market_file['Популярная_категория'].unique()


# In[23]:


market_money.info()


# In[24]:


market_money.isna().mean()


# In[25]:


market_money.duplicated().sum()


# In[26]:


market_money['Период'].unique()


# In[27]:

market_time.info()


# In[28]:


market_time.isna().mean()


# In[29]:


market_time.duplicated().sum()


# In[30]:


market_time['Период'].unique()


# In[31]:


money.info()


# In[32]:


money.isna().mean()


# In[33]:


money.duplicated().sum()



# In[34]:


market_file = market_file.rename(columns={
    'Покупательская активность': 'Покупательская_активность',
    'Тип сервиса': 'Тип_сервиса',
    'Разрешить сообщать': 'Разрешить_сообщать'
})


# In[35]:


market_file.loc[market_file['Тип_сервиса'] == 'стандартт', 'Тип_сервиса'] = 'стандарт'



# In[36]:


market_money.loc[market_money['Период'] == 'препредыдущий_месяц', 'Период'] = 'предыдущий_месяц'


# In[37]:


market_money = market_money.rename(columns={'Период': 'Период_деньги'})



# In[38]:


market_time['Период'] = market_time['Период'].str.replace('предыдцщий_месяц', 'предыдущий_месяц')


# In[39]:


market_time = market_time.rename(columns={'Период': 'Период_время'})



# In[40]:


market_file = market_file.set_index('id')
market_money = market_money.set_index('id')
market_time = market_time.set_index('id')
money = money.set_index('id')


# In[41]:


def plot_boxplots(filename):

    numeric_cols = filename.select_dtypes(include=['number']).columns

    num_plots = len(numeric_cols)
    plt.figure(figsize=(10, 5 * num_plots))

    for i, col in enumerate(numeric_cols):

        plt.subplot(num_plots, 2, 2*i + 2)
        filename.boxplot(column=col)
        plt.title(f'График ящик с усами для целевого признака {col}')

    plt.tight_layout()
    plt.show()


# In[42]:


def plot_histograms_1(filename, n_bins):
    numeric_cols = filename.select_dtypes(include=['number']).columns

    num_plots = len(numeric_cols)
    plt.figure(figsize=(10, 5 * num_plots))

    for i, col in enumerate(numeric_cols):
        plt.subplot(num_plots, 2, 2*i + 1)
        filename[col].hist(bins=15)
        plt.title(f'Столбчатая диаграмма для целевого признака {col}')
        plt.xlabel(col)
        plt.ylabel('Частота')
        
    plt.tight_layout()
    plt.show()



# In[43]:


market_file.describe()


# In[44]:


plot_boxplots(market_file)


# В столбцах:
# - Маркет_актив_6_мес;
# - Акционные_покупки;
# - Неоплаченные_продукты_штук_квартал 
# присутствуют выбросы.

# In[45]:


plot_histograms_1(market_file, 25)



# ### Датафрейм market_money.csv

# #### Описательная статистика (столбчатые диаграммы и графики boxplot) для датафрейма market_money.csv

# In[46]:


market_money.describe()


# In[47]:


plot_boxplots(market_money)


# In[48]:


plot_histograms_1(market_money, 15)


# In[49]:


market_money.sort_values(by='Выручка', ascending=False).head(10)



# In[50]:


market_money = market_money[market_money['Выручка'] != 106862.2]



# In[51]:


plot_boxplots(market_money)


# In[52]:


plot_histograms_1(market_money, 50)


# In[53]:


market_money.sort_values(by='Выручка', ascending=True).head(10)



# In[54]:


market_money = market_money[market_money['Выручка'] != 0]



# In[55]:


market_time.describe()


# In[56]:


plot_boxplots(market_time)


# In[57]:


plot_histograms_1(market_time, 20)


# In[58]:


money.describe()



# In[59]:


plot_boxplots(money)


# In[60]:


plot_histograms_1(money, 20)

 

# In[61]:


market_money.head(5)


# In[62]:


market_time.head(10)


# In[63]:


df_money_grouped = market_money.groupby(['id', 'Период_деньги']).agg({'Выручка': 'sum'}).reset_index()


# In[64]:

# In[65]:


df_time_grouped = market_time.groupby(['id', 'Период_время']).agg({'минут': 'sum'}).reset_index()


# In[66]:

# In[67]:


df_money_temp = pd.DataFrame(columns=['id', 'текущий_месяц_выручка', 'предыдущий_месяц_выручка'])


# In[68]:


df_time_temp = pd.DataFrame(columns=['id', 'предыдущий_месяц_минут', 'текущий_месяц_минут'])


# In[69]:


df_money_test = pd.pivot_table(df_money_grouped, values = 'Выручка', index=['id'], 
                               columns=['Период_деньги'], aggfunc="sum")


# In[70]:


df_money_test.reset_index()


# In[71]:


df_time_test = pd.pivot_table(df_time_grouped, values = 'минут', index=['id'], 
                               columns=['Период_время'], aggfunc="sum")


# In[72]:


df_time_test


# In[73]:


df_combined = pd.merge(df_money_test, df_time_test, on='id')
df = pd.merge(market_file, df_combined, on='id')


# In[74]:


df.columns


# In[75]:


df = df.rename(columns={'предыдущий_месяц_x': 'предыдущий_месяц_деньги'})
df = df.rename(columns={'предыдущий_месяц_y': 'предыдущий_месяц_время'})
df = df.rename(columns={'текущий_месяц_x': 'текущий_месяц_деньги'})
df = df.rename(columns={'текущий_месяц_y': 'текущий_месяц_время'})


# In[76]:


df.shape


# In[77]:


df[df[['предыдущий_месяц_деньги','текущий_месяц_деньги']].eq(0).any(axis=1)].shape[0]


# In[78]:


df = df[df['предыдущий_месяц_деньги'] != 0]


# In[79]:


df = df[df['текущий_месяц_деньги'] != 0]


# In[80]:


df = df.dropna()


# In[81]:


df.shape



# In[82]:


def fitter_function(row_name):
    for_f = df[row_name].values
    f = Fitter(for_f,
           distributions=['gamma',
                          'lognorm',
                          "beta",
                          "burr",
                          "norm"])
    f.fit()
    return f.summary()


# In[83]:


fitter_function('Маркет_актив_тек_мес')


# In[84]:


fitter_function('Маркет_актив_6_мес')


# In[85]:


fitter_function('Длительность')


# In[86]:


fitter_function('Акционные_покупки')


# In[87]:


fitter_function('Средний_просмотр_категорий_за_визит')


# In[88]:


fitter_function('Неоплаченные_продукты_штук_квартал')


# In[89]:


fitter_function('Ошибка_сервиса')


# In[90]:


fitter_function('Страниц_за_визит')



# In[91]:


numeric_cols = df.select_dtypes(include=['number']).columns
correlation_matrix = df[numeric_cols].corr(method='spearman')

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()


# In[92]:


numeric_cols = df.select_dtypes(include=['number']).columns
X = add_constant(df[numeric_cols])
pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)


# In[93]:


RANDOM_STATE = 42
TEST_SIZE = 0.25


# In[94]:


df.head(5)


# In[95]:


encoder = LabelEncoder()

X = df.drop(['Покупательская_активность'], axis=1)
y = df['Покупательская_активность']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = TEST_SIZE, 
    random_state = RANDOM_STATE,
    stratify = y)

ohe_columns = ['Разрешить_сообщать', 
            'Популярная_категория']

ord_columns = ['Тип_сервиса']

num_columns = [
    'Маркет_актив_6_мес', 
    'Маркет_актив_тек_мес', 
    'Длительность',
    'Акционные_покупки', 
    'Средний_просмотр_категорий_за_визит',
    'Неоплаченные_продукты_штук_квартал', 
    'Ошибка_сервиса',
    'Страниц_за_визит', 
    'предыдущий_месяц_деньги', 
    'текущий_месяц_деньги',
    'предыдущий_месяц_время', 
    'текущий_месяц_время']

encoder.fit(y_train)

y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)


# In[96]:


ohe_pipe = Pipeline(
    [
        (
            'simpleImputer_ohe', 
            SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        ),
        (
            'ohe', 
            OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        )
    ]
) 
print(ohe_pipe)


# In[97]:


df['Тип_сервиса'].unique()


# In[98]:


ord_pipe = Pipeline(
    [
        (
            'simpleImputer_before_ord', 
            SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        ),
        (
            'ord',
            OrdinalEncoder(
                categories=[
                    ['премиум', 'стандарт']
                ], 
                handle_unknown='use_encoded_value',
                unknown_value=np.nan
            )
        ),
        (
            'simpleImputer_after_ord', 
            SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        )
    ]
)


# In[99]:


num_pipe = Pipeline(
    [
        (
            'scaler', 
            StandardScaler(),  MinMaxScaler())
    ]
)
 
print(num_pipe)


# In[100]:


data_preprocessor = ColumnTransformer(
    [
        ('ohe', ohe_pipe, ohe_columns),
        ('ord', OrdinalEncoder(), ord_columns),
        ('num', num_pipe, num_columns)
    ], 
    remainder='passthrough'
)   


# In[101]:


data_preprocessor_2 = ColumnTransformer([
    
    ('ohe', OneHotEncoder(drop='first', handle_unknown='error'), ohe_columns),
    ('ord', ord_pipe, ord_columns),
    ('num', MinMaxScaler(), num_columns)
], 
    remainder='passthrough'
)


# In[102]:


pipe_final = Pipeline(
    [
        ('preprocessor', data_preprocessor),
        ('models', DecisionTreeClassifier(random_state=RANDOM_STATE))
    ]
)


# In[103]:


param_grid = [
    {
        'models': [DecisionTreeClassifier(random_state=RANDOM_STATE)],
        'models__max_depth': range(2, 5),
        'models__max_features': range(2, 5),
        'preprocessor': [data_preprocessor, data_preprocessor_2]

    },
    {
        'models': [KNeighborsClassifier()],
        'models__n_neighbors': range(5, 100),
        'preprocessor': [data_preprocessor, data_preprocessor_2]
    },

    {
        'models': [LogisticRegression(random_state=RANDOM_STATE)],
        'models__C': [0.1, 1.0, 10.0, 100.0],
         'preprocessor': [data_preprocessor, data_preprocessor_2]
    },
    {
        'models': [SVC(probability=True, random_state=RANDOM_STATE)],
        'models__kernel': ['linear', 'rbf'],
        'preprocessor': [data_preprocessor, data_preprocessor_2]
    }
]


# In[104]:


randomized_search = RandomizedSearchCV(
    pipe_final, 
    param_grid, 
    cv=5,
    scoring='roc_auc',
    random_state=RANDOM_STATE,
    n_jobs=-1
)


# In[105]:


get_ipython().run_cell_magic('time', '', "randomized_search.fit(X_train, y_train)\nprint('Лучшая модель и её параметры:\\n\\n', randomized_search.best_estimator_)\nprint ('Метрика лучшей модели на тренировочной выборке:', randomized_search.best_score_)")


# In[106]:


y_test_pred = randomized_search.predict(X_test)


# In[107]:


print('Лучшая модель и её параметры:\n\n', randomized_search.best_estimator_)
print ('Метрика лучшей модели на тренировочной выборке:', randomized_search.best_score_)


# In[108]:


best_model = randomized_search.best_estimator_
predictions = best_model.predict_proba(X)[:, 1]


# In[109]:


y_test.shape


# In[110]:


y_test_pred.shape


# In[111]:


print(f'Метрика ROC-AUC на тестовой выборке: {round(roc_auc_score(y_test, y_test_pred),3)}')


# In[112]:


shap.plots.initjs() 
best_model = randomized_search.best_estimator_.named_steps['models']
preprocessor = randomized_search.best_estimator_.named_steps['preprocessor']
X_train_preprocessed = preprocessor.transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)


# In[113]:


preprocessor.fit(df)


# In[114]:


ohe_feature_names = preprocessor.named_transformers_['ohe'].get_feature_names_out(input_features=ohe_columns)

num_feature_names = num_columns

ord_feature_names = ord_columns

all_feature_names = np.concatenate([ohe_feature_names, num_feature_names, ord_feature_names])


# In[115]:


X_train_preprocessed.shape


# In[116]:


X_train_preprocessed_df = pd.DataFrame(X_train_preprocessed, columns=all_feature_names)
X_test_preprocessed_df = pd.DataFrame(X_test_preprocessed, columns=all_feature_names)


# In[117]:


COUNT=20

X_train_preprocessed_shap = shap.sample(X_train_preprocessed_df, 
                                        COUNT, 
                                        random_state=RANDOM_STATE)

X_test_preprocessed_shap = shap.sample(X_test_preprocessed_df, 
                                       COUNT, 
                                       random_state=RANDOM_STATE)


# In[118]:


best_model = randomized_search.best_estimator_.named_steps['models']
preprocessor = randomized_search.best_estimator_.named_steps['preprocessor']


# In[119]:


explainer = shap.KernelExplainer(best_model.predict_proba, X_train_preprocessed_shap)
shap_values = explainer(X_test_preprocessed_shap)


# In[120]:


shap_values[:, :, 1].shape


# In[121]:


plt.title('Feature Importance Bar', fontsize=14)
plt.ylabel('Наименование признаков', fontsize=14)

shap.summary_plot(shap_values[:, :, 1], X_test_preprocessed, 
                  feature_names=all_feature_names, 
                  plot_type='bar')


# In[122]:


shap.summary_plot(shap_values[:, :, 1], X_test_preprocessed_shap)



# In[123]:


threshold_probability = 0.7


# In[124]:


threshold_profit = 3


# In[125]:


df['вероятность снижения'] = predictions


# In[126]:


df_seg = df


# In[127]:


df_seg = df.join(money, on='id')


# In[128]:


df_merge = df.join(money, on='id')


# In[129]:


df_seg.info()


# In[130]:


df_seg_2 = df.join(money, on='id')


# In[131]:


prediction_indicate_1 = np.where(df_seg['Прибыль'] > threshold_profit, 
                               True, False)


# In[132]:


df_seg = df_seg.loc[prediction_indicate_1]


# In[133]:


df_seg.head()


# In[134]:


prediction_indicate_2 = np.where(df_seg['вероятность снижения'] > threshold_probability, 
                               True, False)


# In[135]:


df_seg = df_seg.loc[prediction_indicate_2]



# In[136]:
# In[137]:


plt.figure(figsize=(12,6))
sns.scatterplot(data=df_merge, x='вероятность снижения', y='Прибыль', label='Другие сегменты', color='purple')
sns.scatterplot(data=df_seg, x='вероятность снижения', y='Прибыль', label='Исследуемый сегмент', color='blue')
plt.legend()
plt.axvline(x=threshold_probability, color='green', linestyle = '--')
plt.axhline(y=threshold_profit, color='green', linestyle = '--')
plt.show()


# In[138]:


def plot_top_features_1(n_bins, y, range_left, range_right, ylabel, xlabel, title):
    fig, axs = plt.subplots(1, 2, tight_layout=True)

    N, bins, patches = axs[0].hist(df_seg[y], bins=n_bins, range= (range_left,range_right))

    fracs = N / N.max()

    norm = colors.Normalize(fracs.min(), fracs.max())

    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)

    axs[1].hist(df_seg[y], bins=n_bins, density=True, range= (range_left,range_right))

    axs[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    return plt


# In[139]:


def plot_top_features_2(n_bins, y, range_left, range_right, ylabel, xlabel, title):
    fig, axs = plt.subplots(1, 2, tight_layout=True)

    N, bins, patches = axs[0].hist(df_merge[y], bins=n_bins, range= (range_left,range_right))

    fracs = N / N.max()

    norm = colors.Normalize(fracs.min(), fracs.max())

    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)

    axs[1].hist(df_merge[y], bins=n_bins, density=True, range= (range_left,range_right))

    axs[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    return plt


# In[140]:


df_seg.shape


# In[141]:


df_merge.shape



# In[142]:


df_seg['Акционные_покупки'].describe()


# In[143]:


plot_top_features_1(15, 'Акционные_покупки', 0.000000, 0.990000,  'Кол-во покупателей',
             'Доля', 'Распределение целевого признака Акционные_покупки')
plt.show()


# In[144]:


df_merge['Акционные_покупки'].describe()


# In[145]:


plot_top_features_2(15, 'Акционные_покупки', 0.000000, 0.990000,  'Кол-во покупателей',
             'Доля', 'Распределение целевого признака Акционные_покупки')
plt.show()



# In[146]:


df_seg['Средний_просмотр_категорий_за_визит'].describe()


# In[147]:


plot_top_features_1(15, 'Средний_просмотр_категорий_за_визит', 1.000000, 6.000000,  'Кол-во покупателей',
             'Доля', 'Средний_просмотр_категорий_за_визит')
plt.show()


# In[148]:


df_merge['Средний_просмотр_категорий_за_визит'].describe()


# In[149]:


plot_top_features_2(15, 'Средний_просмотр_категорий_за_визит', 1.000000, 6.000000,  'Кол-во покупателей',
             'Доля', 'Средний_просмотр_категорий_за_визит')
plt.show()


# In[150]:


df_seg['Неоплаченные_продукты_штук_квартал'].describe()


# In[151]:


plot_top_features_1(25, 'Неоплаченные_продукты_штук_квартал', 0.000000, 10.000000,  'Кол-во покупателей',
             'Доля', 'Распределение целевого признака Неоплаченные_продукты_штук_квартал')
plt.show()


# In[152]:


df_merge['Неоплаченные_продукты_штук_квартал'].describe()


# In[153]:


plot_top_features_2(25, 'Неоплаченные_продукты_штук_квартал', 0.000000, 10.000000,  'Кол-во покупателей',
             'Доля', 'Распределение целевого признака Неоплаченные_продукты_штук_квартал')
plt.show()



# In[154]:


df_seg['текущий_месяц_время'].describe()


# In[155]:


plot_top_features_1(25, 'текущий_месяц_время', 4.000000, 17.000000,  'Кол-во покупателей',
             'Доля', 'Распределение целевого признака текущий_месяц_время')
plt.show()


# In[156]:


df_merge['текущий_месяц_время'].describe()


# In[157]:


plot_top_features_2(25, 'текущий_месяц_время', 4.000000, 23.000000,  'Кол-во покупателей',
             'Доля', 'Распределение целевого признака текущий_месяц_время')
plt.show()



# In[158]:


df_seg['предыдущий_месяц_деньги'].describe()


# In[159]:


plot_top_features_1(25, 'предыдущий_месяц_деньги', 7370.000000, 11634.000000,  'Кол-во покупателей',
             'Доля', 'Распределение целевого признака предыдущий_месяц_деньги')
plt.show()


# In[160]:


df_merge['предыдущий_месяц_деньги'].describe()


# In[161]:


plot_top_features_2(25, 'предыдущий_месяц_деньги', 7103.000000, 12209.500000,  'Кол-во покупателей',
             'Доля', 'Распределение целевого признака предыдущий_месяц_деньги')
plt.show()
