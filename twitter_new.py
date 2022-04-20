import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import datetime
from wordcloud import WordCloud,STOPWORDS
import re
import geojson
from dash_fonctions import *
import pickle
import os
import base64

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache(allow_output_mutation=True)
def get_img_with_href(local_img_path, target_url):
    img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_img_path)
    html_code = f'''
        <a href="{target_url}" target="_blank" >
            <img src="data:image/{img_format};base64,{bin_str}" />
        </a>'''
    return html_code


def positions_df(dataframe,level):
	places = []	
	for i in dataframe[level]:
		for j in i:
			places.append(j)
	if level == 'village':
		positions = pd.DataFrame.from_dict(Counter(places), orient='index').reset_index()
		positions.columns=['village','occurences']
	else:
		dico_positions = dict(Counter(places))
		if level == 'district':
			subdivisions = pickle.load(open("districts.p", "rb"))
		else:
			subdivisions = pickle.load(open("regions.p", "rb"))
		for i in subdivisions:
			if i not in dico_positions:
				dico_positions[i]=0
		positions = pd.DataFrame.from_dict(dico_positions, orient='index').reset_index()
		positions.columns = [level, 'occurences']
	
	return positions	
		

def cartographier(dataframe,level):

	positions=positions_df(dataframe,level)
	
	if level == 'village':
		title='Localities mentionned in tweets'
		coordinates = pd.read_csv('maps/P code.csv', decimal=',')		
		positions['longitude'] = positions['village'].apply(
		        lambda x: coordinates[coordinates['NAME'] == x].iloc[0]['X_COORD'])
		positions['latitude'] = positions['village'].apply(
		        lambda x: coordinates[coordinates['NAME'] == x].iloc[0]['Y_COORD'])
		carte = px.scatter_mapbox(positions, lat="latitude", lon="longitude", size='occurences',
                                 zoom=5.5, hover_name='village', hover_data=['occurences'],
                                 color_discrete_sequence=['red'], title='Places mentionned in tweets',width=900, height=1000)
		carte.update_layout(mapbox_style="open-street-map")
		carte.update_layout(margin={"r": 5, "t": 0, "l": 5, "b": 0})
	
	else:
		
		if level == 'district':
			title='Districts mentionned in tweets'
			with open("districts_som.geojson") as f:
				gj = geojson.load(f)
						
					
		else:
			title='Regions mentionned in tweets'
			with open("regions_som.geojson") as f:
				gj = geojson.load(f)
			

		dico = {'region': 'admin1', 'district': 'admin2'}
		detail = dico[level]
		feature_id_dico = {'region': 'properties.ADM1_EN',
		                   'district': 'properties.ADM2_EN'}
	
		carte = px.choropleth_mapbox(positions,  # Input Dataframe
		                    geojson=gj,  # identify country code column
		                    color="occurences",  # identify representing column
	                            hover_name=level,              # identify hover name
	                            locations=level,
	                            opacity=0.5,  # select projection
	                            color_continuous_scale="icefire",  # select prefer color scale
	                            featureidkey=feature_id_dico[level],
	                            range_color=[0, positions['occurences'].max()*1.1],  # select range of dataset
	                            center={"lat": 4.5517, "lon": 45.7073},
	                            zoom=5.5,width=700, height=1000,
	                            )
		carte.update_geos(fitbounds='locations', visible=False)
		carte.update_layout(mapbox_style="open-street-map")
		carte.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

	return carte,title
	

st.set_page_config(layout="wide")

axiom_html = get_img_with_href('logoAxiom.png', 'https://axiom.co.ke')

st.sidebar.markdown(axiom_html, unsafe_allow_html=True)
st.sidebar.title('Analysis of Tweets in Somalia')



with st.sidebar:
    choose = option_menu(None, ["Elections", "Drought",'Security Incidents','Displacements'],
                         icons=["envelope paper", 'sun','emoji-angry','reply fill'],
                         menu_icon="app-indicator", default_index=0,
                        )

a,b=st.columns((3,1))

if choose == "Elections":
    date_begin='25th of March 2022'
    data = pd.read_csv('Elections.csv', sep='\t')
    a.title('Tweets related to elections')
    a.caption('Gather all tweets which include either the word "doorashooyinka" referred to as Somali tweets or the words "election" and "somalia" referred to as english tweets')
elif choose=='Drought':
    date_begin='25th of March 2022'
    data = pd.read_csv('Drought.csv', sep='\t')
    a.title('Tweets related to drought')
    a.caption('Gather all tweets which include either the word "abaaro" referred to as Somali tweets or the words "drought" and "somalia" referred to as english tweets')
elif choose=='Displacements':
    date_begin='31st of March 2022'
    data = pd.read_csv('Displacements.csv', sep='\t')
    a.title('Tweets related to displacements')
    a.caption('Gather all tweets which include either at least one of the words "barakacyaal", "qaxooti" or "barakacayaasha" referred to as Somali tweets or the words "somali" AND one of the words "idp", "refugee", "displaced" referred to as english tweets')
else:
    date_begin='31st of March 2022'
    data = pd.read_csv('Attacks.csv', sep='\t')
    a.title('Tweets related to security incidents')
    a.caption('Gather all tweets which include either at least one of the words "weerar", "qarax","dagaal" or "colaad" referred to as Somali tweets or the words "somalia" AND one of the words "attack", "conflict", "fighting", "explosion" or "bombing" referred to as english tweets')


data['date'] = data['created_at'].apply(lambda x: x[:10])
data['date'] = data['date'].apply(lambda x: pd.to_datetime(x))
data['created_at']=data['created_at'].apply(lambda x: pd.to_datetime(x))


b.title(str(len(data))+ ' tweets')
b.write('since the '+date_begin)

col1,col2=st.columns((1,1))

numbers = draw_numbers(pd.to_datetime("2022-03-25"), pd.to_datetime(str(datetime.datetime.today()).split()[0]), data,
                       'Tweets per day per langage')
numbers_sentiment=draw_sentiments(pd.to_datetime("2022-03-25"), pd.to_datetime(str(datetime.datetime.today()).split()[0]), data,
                       'Tweets per day according to sentiment analysis')                       

col1.plotly_chart(numbers, use_container_width=True)
col2.plotly_chart(numbers_sentiment, use_container_width=True)


for i in ['village','district','region']:
    data[i] = data[i].apply(lambda x : eval(x))


col1,col2=st.columns((7,5))
col1.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
level = col1.radio('Select the level of visualization:', ('Village', 'District', 'Region')).lower()
carte1,title1=cartographier(data,level)
col1.subheader(title1)
col1.plotly_chart(carte1, use_container_width=True)

col2.subheader('Overall sentiment analysis')
pies=sentiment_pies(pd.to_datetime("2022-03-25"), pd.to_datetime(str(datetime.datetime.today()).split()[0]),data)
col2.plotly_chart(pies, use_container_width=True)

col2.subheader('Evolution of number of positive tweets vs negative tweets')
col2.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
langage=col2.radio("Choose which tweets",('All','English','Somali'))

waterf=waterfall(pd.to_datetime("2022-03-25"), pd.to_datetime(str(datetime.datetime.today()).split()[0]),data,langage)
col2.plotly_chart(waterf, use_container_width=True)

st.markdown('---')

if st.checkbox('Select time range and/or specific locations'):

    maintenant = datetime.datetime.now().date()
    debut = datetime.date(year=2022,month=3,day=25)
    values = st.slider(
        'Select a range of time',
        debut, maintenant, (debut, maintenant))

    df = data[(data['date'].apply(lambda x: x.date()) >= values[0]) &
              (data['date'].apply(lambda x: x.date()) <= values[1])].copy()
    st.subheader('There has been a total of '+str(len(df))+ ' between the '+ str(values[0])+ ' and the '+ str(values[1]))
	
    carte2,title2=cartographier(df,level)
    st.subheader(title1)
    st.write('(between the '+ str(values[0])+ ' and the '+ str(values[1])+')')
    st.plotly_chart(carte1, use_container_width=True)
    positions=positions_df(df,level)

    places=st.multiselect('Select '+level.capitalize()+'(s)', positions[level].unique())
    df=df[df[level].apply(lambda x:len(x)>0)].copy()

    if places:
        places_df=pd.DataFrame(columns=data.columns)
        for i in places:
            places_df=places_df.append(df[df[level].apply(lambda x : i in x)])
        titre=' or '.join(places)+' during the selected period.'
    else:
        places_df=df.copy()
        titre=' places during the selected period.'

    st.dataframe(places_df[['created_at', 'text', 'Google_Translation', 'village', 'district', 'region']].\
                 sort_values(by='created_at'))

    
    fig_places = draw_numbers(pd.to_datetime(str(values[0])), pd.to_datetime(str(values[1])),
                           places_df, 'Tweets per day mentionning '+titre)
    st.plotly_chart(fig_places, use_container_width=True)

    if st.checkbox('Displaay Wordclouds'):

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





