#!/usr/bin/env python
# coding: utf-8

# <font color='blue' size=5><b>Комментарий ревьюера</b></font>
# 
# <font color='blue'>Привет, Мария! Меня зовут Павел Григорьев, и я буду проверять этот проект.<br>Моя основная цель - не указать на совершённые тобой ошибки, а поделиться своим опытом и помочь тебе совершенствоваться как профессионалу.<br>Спасибо за проделанную работу! Предлагаю общаться на «ты».</font>
# <details>
# 	<summary><u>Инструкция по организационным моментам (кликабельно)</u>⤵</summary>
# <font color='blue'>Я буду использовать различные цвета, чтобы было удобнее воспринимать мои комментарии:</font>
# 
# 
# ---
# 
# 
# <font color='blue'>синий текст - просто текст комментария</font>
# 
# <font color='green'>✔️ и зеленый текст - все отлично</font>
# 
# <font color='orange'>⚠️ и оранжевый текст - сделано все правильно, однако есть рекомендации, на что стоит обратить внимание</font>
# 
# <font color='red'>❌ и красный текст - есть недочеты</font>
# 
# 
# </details>    
#     </br>
# <font color='blue'>Пожалуйста, не удаляй мои комментарии в случае возврата работы, так будет проще разобраться, какие были недочеты, а также сразу увидеть исправленное. </font>
# 
# Ответы на мои комментарии лучше тоже помечать.
# Например: <font color='purple'><b>Комментарий студента</b></font>
# 
# <font color='blue'><b>Давай смотреть, что получилось!</b></font>

# <font color='blue' size=3><b>Комментарий ревьюера 2</b></font>
# 
# <font color='blue'>Привет еще раз. Спасибо, за исправления. Оформление комментариев по работе сохраняется. Только обозначим, что это вторая итерация.</font>

# <font color='blue' size=3><b>Комментарий ревьюера 3</b></font>
# 
# <font color='blue'>Привет еще раз. Спасибо, за исправления. Оформление комментариев по работе сохраняется.</font>

# # Обучение с учителем: качество модели для Интернет-магазина «В один клик»

# 

# Необходимо проанализировать данные интернет-магазина «В один клик», который продаёт разные товары: для детей, для дома, мелкую бытовую технику, косметику и даже продукты. 
# 
# 
# <b>Задача</b>: разработать решение, которое позволит персонализировать предложения постоянным клиентам, чтобы увеличить их покупательскую активность.
# 
# <b>Цели</b> необходимые для достижения поставленной задачи:
# - Нужно промаркировать уровень финансовой активности постоянных покупателей;
# - Нужно построить модель, которая предскажет вероятность снижения покупательской активности клиента в следующие три месяца;
# - В исследование нужно включить дополнительные данные финансового департамента о прибыльности клиента: какой доход каждый покупатель приносил компании за последние три месяца;
# - Используя данные модели и данные о прибыльности клиентов, нужно выделить сегменты покупателей и разработать для них персонализированные предложения.
# 
# 
# Интернет-магазин «В один клик» предоставил два датафрейма: 
# - Файл market_file.csv, который содержит данные о поведении покупателя на сайте, о коммуникациях с покупателем и его продуктовом поведении. 
# - Файл market_money.csv, который содержит данные о выручке, которую получает магазин с покупателя, то есть сколько покупатель всего потратил за период взаимодействия с сайтом.
# - Файл market_time.csv, который содержит данные о времени (в минутах), которое покупатель провёл на сайте в течение периода.
# - Файл money.csv, который содержит с данными о среднемесячной прибыли покупателя за последние 3 месяца: какую прибыль получает магазин от продаж каждому покупателю.

# <b>Описание данных датафрейма market_file.csv</b> 
# 
# - <b>id</b> — номер покупателя в корпоративной базе данных; (0)
# - <b>Покупательская активность</b> — рассчитанный класс покупательской активности (целевой признак): «снизилась» или «прежний уровень»; (1)
# - <b>Тип сервиса</b> — уровень сервиса, например «премиум» и «стандарт»; (2)
# - <b>Разрешить сообщать</b> — информация о том, можно ли присылать покупателю дополнительные предложения о товаре. Согласие на это даёт покупатель; (3)
# - <b>Маркет_актив_6_мес</b> — среднемесячное значение маркетинговых коммуникаций компании, которое приходилось на покупателя за последние 6 месяцев. Это значение показывает, какое число рассылок, звонков, показов рекламы и прочего приходилось на клиента; (4)
# - <b>Маркет_актив_тек_мес</b> — количество маркетинговых коммуникаций в текущем месяце; (5)
# - <b>Длительность</b> — значение, которое показывает, сколько дней прошло с момента регистрации покупателя на сайте; (6)
# - <b>Акционные_покупки</b> —среднемесячная доля покупок по акции от общего числа покупок за последние 6 месяцев; (7)
# - <b>Популярная_категория</b> —  самая популярная категория товаров у покупателя за последние 6 месяцев; (8)
# - <b>Средний_просмотр_категорий_за_визит</b> — показывает, сколько в среднем категорий покупатель просмотрел за визит в течение последнего месяца; (9)
# - <b>Неоплаченные_продукты_штук_квартал</b> — общее число неоплаченных товаров в корзине за последние 3 месяца; (10)
# - <b>Ошибка_сервиса</b> —  число сбоев, которые коснулись покупателя во время посещения сайта; (11)
# - <b>Страниц_за_визит</b> —  среднее количество страниц, которые просмотрел покупатель за один визит на сайт за последние 3 месяца. (12)
# 
# Всего в датафрейме <b>13 признаков</b>. 

# <b>Описание данных датафрейма market_money.csv</b> 
# 
# - <b>id</b> — номер покупателя в корпоративной базе данных; (0)
# - <b>Период</b> —  название периода, во время которого зафиксирована выручка. Например, 'текущий_месяц' или 'предыдущий_месяц'; (1)
# - <b>Выручка</b> — сумма выручки за период. (2)
#  
# Всего в датафрейме <b>3 признака</b>. 

# <b>Описание данных датафрейма market_time.csv</b> 
# 
# - <b>id</b> — номер покупателя в корпоративной базе данных; (0)
# - <b>Период</b> —  название периода, во время которого зафиксировано общее время.; (1)
# - <b>минут</b> —  значение времени, проведённого на сайте, в минутах.. (2)
#  
# Всего в датафрейме <b>3 признака</b>. 

# <b>Описание данных датафрейма money.csv</b> 
# 
# - <b>id</b> — номер покупателя в корпоративной базе данных; (0)
# - <b>Прибыль</b> — значение прибыли. (1)
# 
# Всего в датафрейме <b>2 признака</b>. 

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


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'> Хорошее оформление импортов! \
# Импорты собраны в одной ячейке, разделены на функциональные группы пустой строкой.</font>

# ## Загрузка данных

# In[11]:


market_file = pd.read_csv('/datasets/market_file.csv', sep = ',', decimal='.')
market_money = pd.read_csv('/datasets/market_money.csv', sep = ',', decimal='.')
market_time = pd.read_csv('/datasets/market_time.csv',  sep = ',', decimal='.')
money = pd.read_csv('/datasets/money.csv',  sep = ';', decimal=',')


# <font color='blue'><b>Комментарий ревьюера : </b></font> ✔️\
# <font color='green'> 👍</font>

# In[12]:


market_file.head(5)


# In[13]:


market_money.head(5)


# In[14]:


market_time.head(5)


# In[15]:


money.head(5)


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'>Данные загружены корректно, первичный осмотр проведен.</font>

# ### Изучим информацию о датафрейме market_file.csv

# In[16]:


market_file.info()


# In[17]:


market_file.isna().mean()


# In[18]:


market_file.duplicated().sum()


# Изучим уникальные значения в столбцах Dtype object:

# In[19]:


market_file['Покупательская активность'].unique()


# In[20]:


market_file['Тип сервиса'].unique()


# In[21]:


market_file['Разрешить сообщать'].unique()


# In[22]:


market_file['Популярная_категория'].unique()


# ### Изучим информацию о датафрейме market_money.csv

# In[23]:


market_money.info()


# In[24]:


market_money.isna().mean()


# In[25]:


market_money.duplicated().sum()


# In[26]:


market_money['Период'].unique()


# ### Изучим информацию о датафрейме market_time.csv

# In[27]:


market_time.info()


# In[28]:


market_time.isna().mean()


# In[29]:


market_time.duplicated().sum()


# Аналогично, изучим уникальные значения в столбцах Dtype object:

# In[30]:


market_time['Период'].unique()


# ### Изучим информацию о датафрейме money.csv

# In[31]:


money.info()


# In[32]:


money.isna().mean()


# In[33]:


money.duplicated().sum()


# <font color='DarkBlue'><b>Предварительный вывод:</b></font><br>
# 
# <font color='DarkGreen'> В разделе 1 проведен предварительный анализ исходных датафреймов. 
# При предварительном анализе данных не обнаружено данных, которые не соответствуют описанию. Кроме того, в данных отсутствуют пропуски и пропущенные значения. Явные дубликаты в данных не были обнаружены. Типы данных корректные.

# ## Предобработка данных

# ### Датафрейм market_file.csv

# In[34]:


market_file = market_file.rename(columns={
    'Покупательская активность': 'Покупательская_активность',
    'Тип сервиса': 'Тип_сервиса',
    'Разрешить сообщать': 'Разрешить_сообщать'
})


# In[35]:


market_file.loc[market_file['Тип_сервиса'] == 'стандартт', 'Тип_сервиса'] = 'стандарт'


# <font color='blue'><b>Комментарий ревьюера : </b></font> ✔️\
# <font color='green'> 👍</font>

# ### Датафрейм market_money.csv

# In[36]:


market_money.loc[market_money['Период'] == 'препредыдущий_месяц', 'Период'] = 'предыдущий_месяц'


# In[37]:


market_money = market_money.rename(columns={'Период': 'Период_деньги'})


# ### Датафрейм market_time.csv¶

# In[38]:


market_time['Период'] = market_time['Период'].str.replace('предыдцщий_месяц', 'предыдущий_месяц')


# In[39]:


market_time = market_time.rename(columns={'Период': 'Период_время'})


# <font color='DarkBlue'><b>Предварительный вывод:</b></font><br>
# 
# <font color='DarkGreen'> В разделе 2 проведена предобработка данных в исходных датафреймов. 
# В ходе предобратотки данных были исправлены опечатки.
#     
# Переименованы для единобразия названия столбцов.    

# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'>Данные загружены корректно, первичный осмотр проведен.</font>

# ## Исследовательский анализ данных 

# Отберем клиентов с покупательской активностью не менее трёх месяцев, то есть таких, которые что-либо покупали в этот период.

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


# ### Датафрейм market_file.csv

# #### Описательная статистика (столбчатые диаграммы и графики boxplot) для датафрейма market_file.csv

# In[43]:


market_file.describe()


# plot_countplots_1(market_file, 20)

# In[44]:


plot_boxplots(market_file)


# В столбцах:
# - Маркет_актив_6_мес;
# - Акционные_покупки;
# - Неоплаченные_продукты_штук_квартал 
# присутствуют выбросы.

# In[45]:


plot_histograms_1(market_file, 25)


# Признак Акционные_покупки представляет собой бимодальное распределение.

# ### Датафрейм market_money.csv

# #### Описательная статистика (столбчатые диаграммы и графики boxplot) для датафрейма market_money.csv

# In[46]:


market_money.describe()


# In[47]:


plot_boxplots(market_money)


# In[48]:


plot_histograms_1(market_money, 15)


# В столбце Выручка также присутсвуют выбросы, из-за чего столбчатая диаграмма неинформативна. Лучше удалить эти выбросы.

# In[49]:


market_money.sort_values(by='Выручка', ascending=False).head(10)


# Удалим выброс 106862.2:

# In[50]:


market_money = market_money[market_money['Выручка'] != 106862.2]


# <font color='blue'><b>Комментарий ревьюера : </b></font> ✔️\
# <font color='green'>Действительно явный выброс.
# Можно удалить, а можно и заполнить, например значением предыдущего месяца.</font>

# Перестроим графики:

# In[51]:


plot_boxplots(market_money)


# In[52]:


plot_histograms_1(market_money, 50)


# In[53]:


market_money.sort_values(by='Выручка', ascending=True).head(10)


# Удалим нули:

# In[54]:


market_money = market_money[market_money['Выручка'] != 0]


# <font color='blue'><b>Комментарий ревьюера : </b></font> ✔️\
# <font color='green'>Неактивные удалены 👌</font>

# ### Датафрейм market_time.csv

# #### Описательная статистика (столбчатые диаграммы и графики boxplot) для датафрейма market_time.csv

# In[55]:


market_time.describe()


# In[56]:


plot_boxplots(market_time)


# In[57]:


plot_histograms_1(market_time, 20)


# ### Датафрейм  money.csv

# Из описания датафрейма money.csv следует, что данная таблица (датафрейм) с ионформацией о среднемесячной прибыли покупателя за последние 3 месяца: какую прибыль получает магазин от продаж каждому покупателю. Таким образом, если в данном датафрейме нет значений равных 0, то все пользователи совершали покупки в течение нужного срока, и могут быть отобранными в когорту клиентов с покупательской активностью не менее трёх месяцев. В противном случае, нужно будет отсеять таковых (имеющим нулевые значения). Проверим это ниже.

# In[58]:


money.describe()


# Минимальное значение равно	0.860000, значит все клиенты что-либо покупали в этот период, и никого не нужно исключать.

# In[59]:


plot_boxplots(money)


# In[60]:


plot_histograms_1(money, 20)


# В признаке Прибыль также присутсвуют выбросы.

# <font color='DarkBlue'><b>Предварительный вывод:</b></font><br>
# 
# <font color='DarkGreen'> В разделе 3 проведен исследовательский анализ данных: построены столбчатые диаграммы и boxplot для всех датафреймов. 
#     
# На основании анализа датафрейма money, были отбераны клиенты с покупательской активностью не менее трёх месяцев, то есть такие, которые что-либо покупали в этот период.  

# <font color='blue'><b>Комментарий ревьюера : </b></font> ✔️\
# <font color='green'>Хорошая работа в части исследования данных: молодец, что используешь функции, а также применяешь подходящие для типов данных инструменты.</font>

# ## Объединение таблиц 

# ### Объединим таблицы market_file.csv, market_money.csv, market_time.csv.  

# In[61]:


market_money.head(5)


# In[62]:


market_time.head(10)


# In[63]:


df_money_grouped = market_money.groupby(['id', 'Период_деньги']).agg({'Выручка': 'sum'}).reset_index()


# In[64]:


#df_money_grouped


# <font color='blue'><b>Комментарий ревьюера 2: </b></font> ⚠️\
# <font color='darkorange'>Датафреймы лучше всегда ограничивать методами .head(), .tail(), sample(). Иначе в выводах ячейки записывается весь Датафрейм, хоть он и не выводится на экран полностью. Такие Аутпуты сильно перегружают тетрадку.</font>

# <div class="alert alert-block alert-info">
# <b>🔄Комментарий студента:</b> 
# <font color='DarkGreen'>👌 Принято. Спасибо!
# </font>
# </div>

# In[65]:


df_time_grouped = market_time.groupby(['id', 'Период_время']).agg({'минут': 'sum'}).reset_index()


# In[66]:


#df_time_grouped


# In[67]:


df_money_temp = pd.DataFrame(columns=['id', 'текущий_месяц_выручка', 'предыдущий_месяц_выручка'])


# In[68]:


df_time_temp = pd.DataFrame(columns=['id', 'предыдущий_месяц_минут', 'текущий_месяц_минут'])


# In[69]:


df_money_test = pd.pivot_table(df_money_grouped, values = 'Выручка', index=['id'], 
                               columns=['Период_деньги'], aggfunc="sum")


# <font color='blue'><b>Комментарий ревьюера 2: </b></font> ❌\
# <font color='red'> ~~Здесь не нужно заполнять пропуски нулями. Пропуски появляются на местах значений которые мы удалили в Шаге Исследовательского анализа данных. Это выручка неактивных пользователей. Заполняя нулём мы возвращаем всё в первоначальное состояние.~~</font>

# <div class="alert alert-block alert-info">
# <b>🔄Комментарий студента:</b> 
# <font color='DarkGreen'>👌 Исправлено.
# </font>
# </div>

# <font color='blue'><b>Комментарий ревьюера 3: </b></font> ✔️\
# <font color='green'> 👍</font>

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


# <font color='DarkBlue'><b>Предварительный вывод:</b></font><br>
# 
# <font color='DarkGreen'> В разделе 4 объединены датафреймы market_file.csv, market_money.csv и market_time.csv.  Данные о прибыли из файла money.csv при моделировании не понадобятся, поэтому они не включены.

# <font color='blue'><b>Комментарий ревьюера: </b></font> ❌\
# <font color='red'> ~~Перед объединением таблиц, таблицы df_money и df_time нужно "развернуть" по id, чтобы один id был только в одной строчке. Всего должно получиться 1296 строк (1297 - если заполнять выброс).\
# Могут пригодиться методы pivot_table() или groupby().~~</font>

# <div class="alert alert-block alert-info">
# <b>🔄Комментарий студента:</b> 
# <font color='DarkGreen'>👌 Исправлено, но я у меня получилось 1300. Не могу понять - почему.
# </font>
# </div>

# In[77]:


# Количество строк с нулевыми значениями.
df[df[['предыдущий_месяц_деньги','текущий_месяц_деньги']].eq(0).any(axis=1)].shape[0]


# In[78]:


df = df[df['предыдущий_месяц_деньги'] != 0]


# In[79]:


df = df[df['текущий_месяц_деньги'] != 0]


# In[80]:


df = df.dropna()


# In[81]:


df.shape


# <font color='blue'><b>Комментарий ревьюера 2: </b></font> ❌\
# <font color='red'> ~~Нужно удалить неактивных покупателей и клиента с аномальной выручкой.~~</font>

# <div class="alert alert-block alert-info">
# <b>🔄Комментарий студента:</b> 
# <font color='DarkGreen'>👌 Исправлено.
# </font>
# </div>

# <font color='blue'><b>Комментарий ревьюера 3: </b></font> ✔️\
# <font color='green'> 👍</font>

# ## Корреляционный анализ 

# ### Проверка на нормальность

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


# Проверка на нормальность слеудющих распределений: - распределение Гаусса - бета-распределение - гамма-распределение - логнормальное распределение - Burr distribution (не нашла русский эквивалент). Думаю, что нам достаточно значения ks_pvalue и проанализируем только первые строки. Тогда: нулевая гипотеза, что распределения совпадают, значит, при p-value < 0.05 отклоняем нулевую гипотезу о нормальности распределения. 
# 
# 
# Везде ks_pvalue <0.05, значит никакое из распределений нормальным не является (что, в целом, наверное, и так видно), поэтому используем далее Спирмена.

# ### Корреляционный анализ признаков в количественной шкале в итоговой таблице для моделирования

# In[91]:


numeric_cols = df.select_dtypes(include=['number']).columns
correlation_matrix = df[numeric_cols].corr(method='spearman')

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()


# <font color='blue'><b>Комментарий ревьюера: </b></font> ⚠️\
# <font color='darkorange'>Хорошим решением будет использовать коэффициент корреляции $\phi_K$ , это передовой способ вычисления коэффициентов корреляции для любых типов признаков и не требует предварительного кодирования. Кроме того, в библиотеке есть визуализация. Стоит учитывать, что этот коэффициент показывает не только линейные связи, но и нелинейные. Более подробно понять вид связи поможет визуализация с помощью диаграмм рассеяния. Ссылка на документацию: https://phik.readthedocs.io/en/latest/
# Пример кода:</font>
# ```python
# from phik.report import plot_correlation_matrix
# from phik import report
# 
# # список интервальных признаков
# interval_cols = [...]
# 
# # вычисление коэффициентов корреляции phi для датафрейма df
# phik_overview = df.phik_matrix(interval_cols=interval_cols)
# 
# # визуализация тепловой карты коэффициентов корреляции
# plot_correlation_matrix(
#     phik_overview.values,
#     x_labels=phik_overview.columns,
#     y_labels=phik_overview.index,
#     title=r"correlation $\phi_K$",
#     fontsize_factor=1.5,
#     figsize=(15, 12)
# )
# ```

# <div class="alert alert-block alert-info">
# <b>🔄Комментарий студента:</b> 
# <font color='DarkGreen'>👌 Спасибо, Павел!
# </font>
# </div>

# ### Анализ признаков на мультиколлинеарность с помощью VIF (Variance Inflation Factor)

# In[92]:


numeric_cols = df.select_dtypes(include=['number']).columns
X = add_constant(df[numeric_cols])
pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)


# Для интерпретации:
# 
# VIF находится в диапазон от единицы до плюс бесконечности. Как правило, при интерпретации показателей variance inflation factor придерживаются следующих принципов:
# 
# - VIF = 1, между признаками отсутствует корреляция
# - 1 < VIF <= 5 — умеренная корреляция
# - 5 < VIF <= 10 — высокая корреляция
# - Более 10 — очень высокая
# 
# https://www.dmitrymakarov.ru/opt/mlr-04/

# Мультиколлинеарности в этой выборке нет.

# <font color='DarkBlue'><b>Предварительный вывод:</b></font><br>
# 
# <font color='DarkGreen'> В разделе 5 проведен корреляционный анализ данных.
#     
#     
# Показано, что умеренно коррелируют между собой следующие признаки:
# - Маркет_актив_6мес и Страниц_за_визит (коэффициент корреляции 0.32);
# - Маркет_актив_6мес и минут (коэффициент корреляции 0.25);
# - Страниц_за_визит и минут (коэффициент корреляции 0.37).
#     
#     
#     Также было показано, что мультиколлинеарности в этой выборке нет.
# 

# <font color='blue'><b>Комментарий ревьюера : </b></font> ✔️\
# <font color='green'> 👍</font>

# ## Использование пайплайнов 

# ### Подготовка данных 

# - Во время подготовки данных используем ColumnTransformer. 
# 
# - Количественные и категориальные признаки обработаем в пайплайне раздельно. 
# 
# - Для кодирования категориальных признаков используем как минимум два кодировщика, для масштабирования количественных — как минимум два скейлера.

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


# <font color='blue'><b>Комментарий ревьюера : </b></font> ❌\
# <font color='red'> ~~Энкодер можно обучать только на обучающих данных.\
# Тестовую выборки только трансформируем.~~</font>

# <div class="alert alert-block alert-info">
# <b>🔄Комментарий студента:</b> 
# <font color='DarkGreen'>👌  Убрала Encoder.
# </font>
# </div>

# <font color='blue'><b>Комментарий ревьюера 2: </b></font> ✔️\
# <font color='green'> 👍</font>

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


# <font color='blue'><b>Комментарий ревьюера: </b></font> ❌\
# <font color='red'> ~~Обрати внимание, 'Популярная_категория' - не может быть порядковым признаком. Он не выстраивается по рангу ни по какому фактору.\
# Некоторые модели могут работать только с OE (деревья), но для Линейки это нужно кодировать в OHE.~~</font>

# <div class="alert alert-block alert-info">
# <b>🔄Комментарий студента:</b> 
# <font color='DarkGreen'>👌  Добавила популярную категорию в Ohe, если я правильно поняла. 
# </font>
# </div>

# <font color='blue'><b>Комментарий ревьюера 2: </b></font> ❌\
# <font color='red'> ~~По заданию нужно два Энкодера.\
# Можно передавать в параметрах для LR, KNN и SVC - OHE, а для Дерева OE,\
# a можно засунуть в ord_columns какой-нибудь бинарный признак, например 'тип_сервиса'.~~ </font>

# <div class="alert alert-block alert-info">
# <b>🔄Комментарий студента:</b> 
# <font color='DarkGreen'>👌 Сделала.
# </font>
# </div>

# <font color='blue'><b>Комментарий ревьюера 3: </b></font> ✔️\
# <font color='green'> 👍</font>

# ### Обучение моделей и подборов параметров

# - Обучим четыре модели: KNeighborsClassifier(), DecisionTreeClassifier(), LogisticRegression() и  SVC(). 
# 
# - Для каждой из них подберем как минимум один гиперпараметр, использовав подходящую для задачи метрику.

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


# <font color='blue'><b>Комментарий ревьюера : </b></font> ❌\
# <font color='red'> ~~По заданию нужно два Скалера.\
# Нужно сделать перебор скалеров в 'preprocessor__num'. ~~</font>

# <font color='blue'><b>Комментарий ревьюера: </b></font> ❌\
# <font color='red'> ~~В SVC, тоже, нужно передать random_state.~~</font>

# <font color='blue'><b>Комментарий ревьюера 2: </b></font> ✔️\
# <font color='green'> 👍</font>

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


# <font color='DarkBlue'><b>Предварительный вывод:</b></font><br>
# 
# <font color='DarkGreen'> В разделе 6 обучены четыре модели: KNeighborsClassifier(), DecisionTreeClassifier(), LogisticRegression() и SVC(). С помощью пайплайнов были перебраны варианты масштабирования, модели и их гиперпараметры, чтобы выбрать лучшую комбинацию. Лучшая модель LogisticRegression(C=0.1, random_state=42).

# <font color='blue'><b>Комментарий ревьюера: </b></font> ❌\
# <font color='red'> ~~Нужно аргументировать выбор метрики.~~</font>

# <div class="alert alert-block alert-info">
# <b>🔄Комментарий студента:</b> 
# <font color='DarkGreen'>👌  ROC AUC метрика хорошо подходит для моделей, используемых в задачах бинарной классификации, где необходимо оценить качество модели без привязки к конкретному порогу классификации, когда у нас есть несбалансированные классы или когда важно контролировать баланс между True Positive Rate и False Positive. Rate.
# </font>
# </div>

# <font color='blue'><b>Комментарий ревьюера 2: </b></font> ✔️\
# <font color='green'> 👍</font>

# <font color='blue'><b>Комментарий ревьюера 2: </b></font> ❌\
# <font color='red'> ~~Нужно посчитать  ROC-AUC на тестовых данных.~~</font>

# <div class="alert alert-block alert-info">
# <b>🔄Комментарий студента:</b> 
# <font color='DarkGreen'>👌 Посчитано ниже.
# </font>
# </div>

# In[109]:


y_test.shape


# In[110]:


y_test_pred.shape


# In[111]:


print(f'Метрика ROC-AUC на тестовой выборке: {round(roc_auc_score(y_test, y_test_pred),3)}')


# <font color='blue'><b>Комментарий ревьюера 3: </b></font> ❌\
# <font color='blue'>roc_auc нужно считать по вероятностям классов
# ```python
# probabilities = model.predict_proba(features)
# probabilities_one = probabilities[:, 1]
# print('Площадь ROC-кривой:', roc_auc_score(target, probabilities_one)
# ```
# Самое подробное объяснение метрики, что я видел на русском языке, можно посмотреть здесь: https://alexanderdyakonov.wordpress.com/2017/07/28/auc-roc-площадь-под-кривой-ошибок \
# А внутри кросс-валидации мы просто указываем метрику ROC_AUC и там никаких вероятностей нам не нужно. Так как эта метрика не дифференцируема на всем промежутке (кажется, разрыв в нуле), то там разработчики кросс-валидации хитро меняют ее на эквивалент Gini, так как Gini = 2 ×AUC_ROC – 1 (можно почитать по ссылке выше)
# </font>

# ## Анализ важности признаков

# ### Оценим важность признаков для лучшей модели и построим график важности с помощью метода SHAP

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


# <font color='DarkBlue'><b>Предварительный вывод:</b></font><br>
# 
# <font color='DarkGreen'> В разделе 7 проведен анализ важности признаков. Сделаем выводы о значимости признаков:
# - Длительность, Популярная_категория_кухонная_посуда и предыдущий_Месяц_время - признаки, которые мало значимы для модели;
# - Средний_просмотр_категорий_за_визит, Неоплаченные_Продукты, текущий_месяц_время - признаки, которые сильнее всего влияют на целевой признак;
# 
#     
#    SHAP может помочь решение модели, так что можно использовать эти наблюдения при моделировании и принятии бизнес-решений.

# <font color='blue'><b>Комментарий ревьюера: </b></font> ❌\
# <font color='red'> ~~Поскольку использован метод predict_proba, shap_values имеет две колонки для каждого класса. Для корректного графика нужен срез массива по целевому классу (1).~~</font>

# <font color='blue'><b>Комментарий ревьюера 2: </b></font> ✔️\
# <font color='green'> 👍</font>

# ## Сегментация покупателей
# 

# ### Выполним сегментацию покупателей. Используем результаты моделирования и данные о прибыльности покупателей.

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


# <font color='blue'><b>Комментарий ревьюера : </b></font> ✔️\
# <font color='green'> 👍</font>

# In[136]:


#plt.scatter(predictions, money['Прибыль'], c=prediction_indicate, cmap='coolwarm') 
#plt.plot(7*[threshold], np.arange(1,8), c = 'green', alpha=1)
#plt.title('Сегментация покупателей')
#plt.show()


# In[137]:


plt.figure(figsize=(12,6))
sns.scatterplot(data=df_merge, x='вероятность снижения', y='Прибыль', label='Другие сегменты', color='purple')
sns.scatterplot(data=df_seg, x='вероятность снижения', y='Прибыль', label='Исследуемый сегмент', color='blue')
plt.legend()
plt.axvline(x=threshold_probability, color='green', linestyle = '--')
plt.axhline(y=threshold_profit, color='green', linestyle = '--')
plt.show()

<font color='blue'><b>Комментарий ревьюера 2: </b></font> ❌\
<font color='red'> Нужно выделить сегмент по двум переменным (вероятность снижения и прибыль), для этого определяем и аргументируем два порога по этим осям, проще всего это посмотреть на рассеивание.
![](https://i.postimg.cc/DZ3GSqwK/102458.png)
Это, условно, те с кем имеет смысл работать.\
Затем определяем группу клиентов из сегмента (например по акционным покупкам) и сравниваем этих клиентов со всеми остальньнымим по другим факторам, для выявления различий (можно строить парные графики или накладывать гистограммы). На основании сравнения формируем рекомендации.</font>
# <div class="alert alert-block alert-info">
# <b>🔄Комментарий студента:</b> 
# <font color='DarkGreen'> ? Тут не до конца получилось. Не понимаю - почему.
# </font>
# </div>

# <font color='blue'><b>Комментарий ревьюера 3: </b></font> \
# <font color='blue'> Здесь точки полной таблицы закрывают сегмент. Чтобы было видно разные цвета, нужно сегмент рисовать последним.\
# НО лучше сравнивать не с полной таблицей, а только с пользователями которые не вошли в сегмент.</font>
# ```python
# others = df_merge[~df_merge.index.isin(df_seg.index)]
# ```

# ### Выберем группу покупателей и предложим, как увеличить её покупательскую активность
<font color='blue'><b>Комментарий ревьюера : </b></font> ❌

```python
8.1 Выполните сегментацию покупателей. Используйте результаты моделирования и данные о прибыльности покупателей.
```
<font color='red'> Попробую описать, что мы должны сделать:

Самое важное − мы должны по заданию выбрать некоторый сегмент пользователей, обосновать выбор сегмента, обосновать то, как мы этот сегмент определяем (почему выбираем такие значения признаков для отбора пользователей в сегмент), а дальше исследовать только этот сегмент.

При выделении сегмента нужно опираться на две шкалы:
- вероятность снижения, предикты лучшей модели полученые методом predict_proba() по итересующему нас классу ("Снизилось"), и
- Прибыль от клиентов из таблицы money.csv
    
Границы сегмента по этим двум шкалам можно определить самостоятельно, например по диаграмме рассеяния, где по одной оси будет результат моделирования (вероятность снижения), а по другой прибыль.
    
Далее проводим Анализ данных по выбраному сегменту и какой нибудь группы клиентов, например, которые посещают мало страниц.\
Формулируем предложение по работе с сегментом для увеличения покупательской активности.</font>
# В разделе 7 были выявлены Cредний_просмотр_категорий_за_визит, Неоплаченные_продукты_штук_квартал, Акционные_покупки, предыдущий_месяц_время, текущий_месяц_деньги - признаки, которые сильнее всего влияют на целевой признак, используем их здесь. Также используем Акционные покупки.

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


# #### Сравним Акционные_покупки для пользователей из df_seg и всех пользователей из df_merge

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


# Доля покупателей из выбранного сегмента заинтересована в Акционных покупках, поэтому, возможно, стоит делать больше персональных акций для них.

# #### Сравним Средний_просмотр_категорий_за_визит для пользователей из df_seg и всех пользователей из df_merge

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


# Доля покупателей из выбранного сегмента заинтересована просматривает, в среднем, меньше страниц за один визит.

# #### Сравним Неоплаченные_продукты_штук_квартал для пользователей из df_seg и всех пользователей из df_merge

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


# Доля покупателей из выбранного сегмента имеет больше неоплаченных продуктов за квартал, возможно, стоит предложить им персональные предложения в рассрочку.

# #### Сравним текущий_месяц_время для пользователей из df_seg и всех пользователей из df_merge

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


# #### Сравним предыдущий_месяц_деньги для пользователей из df_seg и всех пользователей из df_merge

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


# <font color='blue'><b>Комментарий ревьюера: </b></font> ❌\
#  <font color='red'>~~Нужен вывод по исследованию Сегмента.~~</font>

# <font color='DarkBlue'><b>Предварительный вывод:</b></font><br>
# 
# <font color='DarkGreen'> роведено графическое и аналитическое исследование группы покупателей. Таким образом, при снижении покупательской активности стоит акцентировать внимание пользователей на акционные товары (Акционные_покупки) в категориях, на просмотр которых уделяется около 10 минут, а также пользователям, которые просматривают за визит около 5 страниц.<font color='red'>

# <font color='blue'><b>Комментарий ревьюера 3: </b></font> ✔️\
# <font color='green'> 👍</font>

# Финальный Вывод:
# 
# Для достижения поставленной задачи по разработке решения, которое позволит персонализировать предложения постоянным клиентам, чтобы увеличить их покупательскую активность, были решены следующие задачи:
# 
# 
# 1) Построена модель, которая предскажет вероятность снижения покупательской активности клиента в следующие три месяца.
# 2) В исследование включены дополнительные данные финансового департамента о прибыльности клиента: какой доход каждый покупатель приносил компании за последние три месяца.
# 3) Используя данные модели и данные о прибыльности клиентов, выделены сегменты покупателей и разработаны для них персонализированные предложения.
# 
# 
# 
#     Для этого были проделаны следующие шаги:
#     
# 1) В разделе 1 проведен предварительный анализ исходных датафреймов.
# 
# 2) В разделе 2 проведена предобработка данных в исходных датафреймов: ходе предобратотки данных были исправлены опечатки. Переименованы для единобразия названия столбцов..
# 
# 3) В разделе 3 проведен исследовательский анализ данных, и именно: были построены графики для каждого признака (распредения и диаграммы размаха для количественных признаков).
#     
# 
# 4) В разделе 4 проведено объедининеие таблицы market_file.csv, market_money.csv, market_time.csv. Данные о прибыли из файла money.csv при моделировании вам не учитывались. 
# 
# 5) В разделе 5 проведен корреляционный анализ признаков в количественной шкале в итоговой таблице для моделирования. 
#     
#     Показано, что умеренно коррелируют между собой следующие признаки:
# 
# - Маркет_актив_6мес и Страниц_за_визит (коэффициент корреляции 0.32);
# - Маркет_актив_6мес и минут (коэффициент корреляции 0.25);
# - Страниц_за_визит и минут (коэффициент корреляции 0.37);
# 
# Также было показано, что мультиколлинеарности в этой выборке нет.
# 
# 6) В разделе 6 обучены четыре модели: KNeighborsClassifier(), DecisionTreeClassifier(), LogisticRegression() и  SVC(). Выберана лучшую модель с использованием заданной метрики -  LogisticRegression(C=0.1, random_state=42).
# 
# 7)  В разделе 7 проведен анализ важности признаков. Сделаем выводы о значимости признаков:
# - Длительность, Популярная_категория_кухонная_посуда и предыдущий_Месяц_время - признаки, которые мало значимы для модели;
# - Средний_просмотр_категорий_за_визит, Неоплаченные_Продукты, текущий_месяц_время - признаки, которые сильнее всего влияют на целевой признак;
#     
#    SHAP может помочь решение модели, так что можно использовать эти наблюдения при моделировании и принятии бизнес-решений.
# 
# 8) В разделе 8 выполнена сегментация покупателей. Проведено графическое и аналитическое исследование группы покупателей. Таким образом, при снижении покупательской активности стоит акцентировать внимание пользователей на акционные товары (Акционные_покупки) в категориях, на просмотр которых уделяется около 10 минут, а также пользователям, которые просматривают за визит около 5 страниц.

# <font color='blue'><b>Комментарий ревьюера : </b></font> ✔️\
# <font color='green'> Очень приятно видеть вывод в конце проекта!\
# Приведены ответы на главные вопросы проекта.</font>

# <font color='blue'><b>Итоговый комментарий ревьюера</b></font>\
# <font color='green'>Мария, хороший проект получился!
# Большое спасибо за проделанную работу. Видно, что приложено много усилий.
# Выводы и рассуждения получились содержательными, их было интересно читать.
# </font>
# 
# <font color='steelblue'>Над проектом ещё стоит поработать - есть рекомендации по дополнению некоторых твоих шагов проекта. Такие рекомендации я отметил жёлтыми комментариями. Будет здорово, если ты учтёшь их - так проект станет структурно и содержательно более совершенным.
# 
# Также в работе есть критические замечания. К этим замечаниям я оставил пояснительные комментарии красного цвета, в которых перечислил возможные варианты дальнейших действий. Уверен, ты быстро с этим управишься:)
# 
# Если о том, что нужно сделать в рамках комментариев, будут возникать вопросы - оставь их, пожалуйста, в комментариях, и я отвечу на них во время следующего ревью.
# 
# Также буду рад ответить на любые твои вопросы по проекту или на какие-либо другие, если они у тебя имеются - оставь их в комментариях, и я постараюсь ответить:)</font>
# 
# <font color='blue'><b>Жду твой проект на повторном ревью. До встречи :) </b></font>

# <font color='blue'><b>Итоговый комментарий ревьюера</b></font>\
# <font color='green'>Мария, большое спасибо за доработу.
# </font>
# 
# <font color='steelblue'>Над проектом ещё стоит поработать - есть рекомендации по дополнению некоторых твоих шагов проекта. Такие рекомендации я отметил жёлтыми комментариями. Будет здорово, если ты учтёшь их - так проект станет структурно и содержательно более совершенным.
# 
# Также в работе есть критические замечания. К этим замечаниям я оставил пояснительные комментарии красного цвета, в которых перечислил возможные варианты дальнейших действий. Уверен, ты быстро с этим управишься:)
# 
# Если о том, что нужно сделать в рамках комментариев, будут возникать вопросы - оставь их, пожалуйста, в комментариях, и я отвечу на них во время следующего ревью.
# 
# Также буду рад ответить на любые твои вопросы по проекту или на какие-либо другие, если они у тебя имеются - оставь их в комментариях, и я постараюсь ответить:)</font>
# 
# <font color='blue'><b>Жду твой проект на повторном ревью. До встречи :) </b></font>

# <font color='blue'><b>Итоговый комментарий ревьюера 3</b></font>\
# <font color='green'> Мария, проект принят! \
# Все этапы пройдены. Все рекомендации учтены.\
# Надеюсь, тебе понравился процесс выполнения и результат.</font> \
# <font color='blue'><b>Спасибо, удачи в освоении профессии!</b></font>
