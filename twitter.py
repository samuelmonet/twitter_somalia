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
import geojson
from dash_fonctions import *
import pickle


st.set_page_config(layout="wide")

with st.sidebar:
    choose = option_menu(None, ["Elections", "Drought",'Security Incidents','Displacements'],
                         icons=["envelope paper", 'sun','emoji angry','reply fill'],
                         menu_icon="app-indicator", default_index=0,
                        )

st.sidebar.title(choose)

if choose == "Elections":
    st.sidebar.write('Elections')
    data = pd.read_csv('Election_all_noRT.csv', sep='\t')
elif choose=='Drought':
    st.title("Tweets on drought")
    data = pd.read_csv('Drought_all_noRT.csv', sep='\t')
elif choose=='Displacements':
    st.title("Tweets on displacements")
    data = pd.read_csv('Displacements_all_noRT.csv', sep='\t')
elif choose=='Security Incidents':
    st.title("Tweets on security incidents")
    data = pd.read_csv('Attacks_all_noRT.csv', sep='\t')
else:
    data = pd.read_csv('test.csv', sep='\t')

data['created_at'] = data['created_at'].apply(lambda x: x[:10])
data['created_at'] = data['created_at'].apply(lambda x: pd.to_datetime(x))

fig = draw_numbers(pd.to_datetime("2022-03-25"), pd.to_datetime(str(datetime.datetime.today()).split()[0]), data,
                       'Tweets per day')

st.plotly_chart(fig, use_container_width=True)

level = st.sidebar.selectbox('Choose the level of visualization:', ['Village', 'District', 'Region']).lower()

for i in ['village','district','region']:
    data[i] = data[i].apply(lambda x : eval(x))

places = []
for i in data[level]:
    for j in i:
        places.append(j)

col1, col2 = st.columns((4, 3))
if level == 'village':
    coordinates = pd.read_csv('maps/P code.csv', decimal=',')
    positions = pd.DataFrame.from_dict(Counter(places), orient='index').reset_index()
    positions.columns=['village','occurences']
    # st.write(villages)
    positions['longitude'] = positions['village'].apply(
        lambda x: coordinates[coordinates['NAME'] == x].iloc[0]['X_COORD'])
    positions['latitude'] = positions['village'].apply(
        lambda x: coordinates[coordinates['NAME'] == x].iloc[0]['Y_COORD'])

    carte = px.scatter_mapbox(positions, lat="latitude", lon="longitude", size='occurences',
                                  zoom=5.5, hover_name='village', hover_data=['occurences'],
                                  color_discrete_sequence=['red'], title='Places mentionned in tweets',width=900, height=1000)
    carte.update_layout(mapbox_style="open-street-map")
    carte.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.subheader('Localities mentionned in tweets')
    st.plotly_chart(carte)

else:
    dico_positions = dict(Counter(places))
    if level == 'district':
        with open("districts_som.geojson") as f:
            gj = geojson.load(f)
        districts = pickle.load(open("districts.p", "rb"))
        for i in districts:
            if i not in dico_positions:
                dico_positions[i]=0
    else:
        with open("regions_som.geojson") as f:
            gj = geojson.load(f)
        regions = pickle.load(open("regions.p", "rb"))
        for i in regions:
            if i not in dico_positions:
                dico_positions[i]=0

    dico = {'region': 'admin1', 'district': 'admin2'}
    detail = dico[level]
    feature_id_dico = {'region': 'properties.ADM1_EN',
                       'district': 'properties.ADM2_EN'}

        # st.write(event_list)
    col1.title('Number of tweets per' + level.capitalize())

    positions = pd.DataFrame.from_dict(dico_positions, orient='index').reset_index()
    positions.columns = [level, 'occurences']
    st.write(positions)
        # st.write(data)
        # st.write(data['value'].max())

    fig = px.choropleth_mapbox(positions,  # Input Dataframe
                            geojson=gj,  # identify country code column
                            color="occurences",  # identify representing column
                            hover_name=level,              # identify hover name
                            locations=level,
                            opacity=0.5,  # select projection
                            color_continuous_scale="icefire",  # select prefer color scale
                            featureidkey=feature_id_dico[level],
                            range_color=[0, positions['occurences'].max()],  # select range of dataset
                            center={"lat": 4.5517, "lon": 45.7073},
                            zoom=5.5,width=700, height=1000,
                            )
    fig.update_geos(fitbounds='locations', visible=False)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)

if st.checkbox('Analyse specific locations'):
    places=st.multiselect('Select '+level.capitalize()+'(s)', positions[level].unique())
    df=data[data[level].apply(lambda x:len(x)>0)].copy()
    places_df=pd.DataFrame(columns=data.columns)
    for i in places:
        places_df=places_df.append(df[df[level].apply(lambda x: i in x)])

    fig_places = draw_numbers(pd.to_datetime("2022-03-25"), pd.to_datetime(str(datetime.datetime.today()).split()[0]),
                           places_df, 'Tweets per day mentionning places in '+ ' or '.join(places))
    st.plotly_chart(fig_places, use_container_width=True)

    st.title('Wordclouds for tweets mentioning locations in '+ ' or '.join(places))
    col1,col2=st.columns((1,1))
    sw = STOPWORDS
    sw.add('t')
    sw.add('https')
    sw.add('co')
    sw.add('rt')

    corpus = ' '.join(places_df[places_df['Google_Translation'].isna()]['text'])
    corpus = re.sub('[^A-Za-z ]', ' ', corpus)
    corpus = re.sub('\s+', ' ', corpus)
    corpus = corpus.lower()
    col1.subheader('English Tweets')
    col1.subheader('')
    col1.subheader('')
    col1.write('')
    col1.caption('')
    col1.caption('')
    sw2 = col1.multiselect('Select words you would like to remove from the wordclouds \n\n',
                               [i[0] for i in Counter(corpus.split(' ')).most_common() if i[0] not in sw][:20])

    for i in sw2:
        sw.add(i)

    if corpus == ' ' or corpus == '':
        corpus = 'Nothing_to_display'
    else:
        corpus = ' '.join([i for i in corpus.split(' ') if i not in sw])
    wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
    wc.generate(corpus)
    col1.image(wc.to_array(), use_column_width=True)

    col2.subheader('Somali Tweets')
    langue = col2.radio("Choose your language",('English','Somali'))
    if langue == 'Somali':
        sw_som = STOPWORDS
        corpus2 = ' '.join(places_df[~(places_df['Google_Translation'].isna())]['Google_Translation'])
    else:
        sw_som=set()
        corpus2 = ' '.join(places_df[~(places_df['Google_Translation'].isna())]['text'])
    corpus2 = re.sub('[^A-Za-z ]', ' ', corpus2)
    corpus2 = re.sub('\s+', ' ', corpus2)
    corpus2 = corpus2.lower()

    sw_som.add('t')
    sw_som.add('https')
    sw_som.add('co')
    sw_som.add('rt')
    sw_som2 = col2.multiselect('Select words you would like to remove from these wordclouds',
                            [i[0] for i in Counter(corpus2.split(' ')).most_common() if i[0] not in sw_som][:20])
    for i in sw_som2:
        sw_som.add(i)
    if corpus2 == ' ' or corpus2 == '':
        corpus2 = 'Nothing_to_display'
    else:
        corpus2 = ' '.join([i for i in corpus2.split(' ') if i not in sw_som])

    wc2 = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
    wc2.generate(corpus2)
    col2.image(wc2.to_array(), use_column_width=True)

if st.checkbox('Display Tweets'):
    st.table(places_df[['created_at','text','Google_Translation','village','district','region']])



