import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import datetime
from wordcloud import WordCloud,STOPWORDS
import re

def draw_numbers(since, until, df, titre):
    d = df.groupby('created_at').aggregate({'created_at': 'count'})
    # st.write(d)
    legend, tickes = echelle(since, until)
    X = [k for k in daterange(since, until)]
    x = [str(k) for k in daterange(since, until)]
    y = [d.loc[i].values[0] if i in d.index else 0 for i in x]
    # st.write(y)
    fig = go.Figure(go.Bar(name='Tweets per day', x=X, y=y))
    fig.add_trace(go.Scatter(name='Week mean number', x=X, y=moyenne(y), mode='lines'))

    # fig.update_layout(autosize=True,legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="center",x=0.5))
    # fig.update_xaxes(ticktext=legend,tickvals=tickes)
    fig.update_layout(title=titre,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=15),
                                  title=dict(font=dict(size=15))))

    return fig


def echelle(since, until):
    start = since.month
    end = until.day
    months = 12 * (until.year - since.year) + until.month - since.month
    year = since.year
    month = since.month
    legend = [calendar[start] + str(year)[-2:]]

    for i in range(months):
        legend.append('')
        month += 1
        if month > 12:
            month = 1
            year += 1
        legend.append(calendar[month] + str(year)[-2:])
    if end > 15:
        legend.append('')

    ticks = ['' for i in range(len(legend))]
    year = since.year
    month = start
    for i in range(len(ticks) // 2):
        ticks[2*i] = datetime.date(year, month, 15)
        if len(ticks) > 2 * i + 1:
            ticks[2*i+1] = datetime.date(year, month, calendar2[legend[2 * i][:-2]])
        ticks[-1] = datetime.date(until.year, until.month, until.day)
        month += 1
        if month > 12:
            month = 1
            year += 1
    # traiter les années bisextiles
    return legend, ticks


def moyenne(liste):
    if len(liste) <= 14:
        return liste
    else:
        L = [0 for i in range(len(liste))]
        for i in range(7):
            j = i + 1
            somme = sum([liste[i + k] for k in range(8)])
            somme2 = sum([liste[-(j + k)] for k in range(8)])
            for k in range(i):
                somme += liste[i - k - 1]
                somme2 += liste[-j + k + 1]
            L[i] = somme / (8 + i)
            L[-j] = somme2 / (8 + i)
        for i in range(7, len(liste) - 7):
            L[i] = sum([liste[i - 7 + k] for k in range(14)]) / 14
        return L


def daterange(date1, date2):
    for n in range(int((date2 - date1).days) + 1):
        yield date1 + datetime.timedelta(n)


def check_words(x, words):
    for word in words:
        if word in x:
            return True
    return False


calendar = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'Mai', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
calendar2 = {'Jan': 31, 'Feb': 28, 'Feb2': 29, 'Mar': 31, 'Apr': 30, 'Mai': 31, 'Jun': 30,
             'Jul': 31, 'Aug': 31, 'Sep': 30, 'Oct': 31, 'Nov': 30, 'Dec': 31}


x, y = np.ogrid[100:500, :600]
mask = ((x - 300)/2) ** 2 + ((y - 300)/3) ** 2 > 100 ** 2
mask = 255 * mask.astype(int)

st.set_page_config(layout="wide")

with st.sidebar:
    choose = option_menu(None, ["Elections", "Drought"],
                         icons=["envelope paper", 'sun'],
                         menu_icon="app-indicator", default_index=0,
                        )


st.sidebar.title(choose)

if choose == "Elections":
    st.sidebar.write('Elections')





elif choose=='Drought':

    st.title("Tweets on drought")




    data = pd.read_csv('test.csv',sep='\t')
    data['created_at'] = data['created_at'].apply(lambda x: x[:10])
    data['created_at'] = data['created_at'].apply(lambda x: pd.to_datetime(x))

    fig = draw_numbers(pd.to_datetime("2022-03-25"), pd.to_datetime(str(datetime.datetime.today()).split()[0]), data,
                       'Tweets per day')

    st.plotly_chart(fig, use_container_width=True)

    level = st.sidebar.selectbox('Choose the level of visualization:', ['Village', 'District', 'Region'])

    correspondances = {'Village':'villages3'}

    for i in ['villages3']:
        data[i] = data[i].apply(lambda x : eval(x))

    places = []
    for i in data[correspondances[level]]:

        for j in i:
            places.append(j)
    if level == 'Village':
        coordinates = pd.read_csv('maps/P code.csv', decimal=',')
        villages = pd.DataFrame.from_dict(Counter(places), orient='index').reset_index()
        villages.columns=['village','occurences']
        # st.write(villages)
        villages['longitude'] = villages['village'].apply(
            lambda x: coordinates[coordinates['NAME'] == x].iloc[0]['X_COORD'])
        villages['latitude'] = villages['village'].apply(
            lambda x: coordinates[coordinates['NAME'] == x].iloc[0]['Y_COORD'])

    col1,col2=st.columns((4,3))

    carte = px.scatter_mapbox(villages, lat="latitude", lon="longitude", size='occurences',
                                zoom=5.5, hover_name='village', hover_data=['occurences'],
                                color_discrete_sequence=['red'], title='Places mentionned in tweets',width=700, height=1000)
    carte.update_layout(mapbox_style="open-street-map")
    carte.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    col1.subheader('Localities mentionned in tweets')
    col1.plotly_chart(carte)

    col2.subheader('Wordclouds')
    sw = STOPWORDS
    sw.add('t')
    sw.add('https')
    sw.add('co')



    corpus = ' '.join(data['text'])
    corpus = re.sub('[^A-Za-z ]', ' ', corpus)
    corpus = re.sub('\s+', ' ', corpus)
    corpus = corpus.lower()

    sw2 = col2.multiselect('Select words you would like to remove from the wordclouds \n\n',
                         [i[0] for i in Counter(corpus.split(' ')).most_common() if i[0] not in sw][:20])

    for i in sw2:
        sw.add(i)

    if corpus == ' ' or corpus == '':
        corpus = 'Nothing to display'
    else:
        corpus = ' '.join([i for i in corpus.split(' ') if i not in sw])
    wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
    wc.generate(corpus)
    col2.subheader('Somalian (A revoir)')
    col2.image(wc.to_array(), use_column_width=True)

    corpus = ' '.join(data['text'])
    corpus = re.sub('[^A-Za-z ]', ' ', corpus)
    corpus = re.sub('\s+', ' ', corpus)
    corpus = corpus.lower()
    if corpus == ' ' or corpus == '':
        corpus = 'Nothing to display'
    else:
        corpus = ' '.join([i for i in corpus.split(' ') if i not in sw])
    wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
    wc.generate(corpus)
    col2.subheader('English')
    col2.image(wc.to_array(), use_column_width=True)

    if st.checkbox('Display Tweets'):
        col1,col2,col3=st.columns((1,1,1))
        words=col1.multiselect('Select words', [i[0] for i in Counter(corpus.split(' ')).most_common() if i[0] not in sw][:50])
        location=col3.multiselect('Select locations', [i[0] for i in Counter(corpus.split(' ')).most_common() if i[0] not in sw][:10])

        st.write(location)
        st.write("les détails des tweets s'affichent ici")

else:

    st.title('')




