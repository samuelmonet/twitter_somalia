import numpy as np
import plotly.graph_objects as go
import datetime
from plotly.subplots import make_subplots

def sentiment_pies(since,until,data):
    df=data[(data.created_at.apply(lambda x: x.date())>=since.date())&\
            (data.created_at.apply(lambda x: x.date())<=until.date())].copy()
    all_tweets=[df.groupby(['date','sentiment']).aggregate({'date': 'count'}).unstack().fillna(0).sum()['date',i]\
              for i in ['positive','neutral','negative']]
    som_tweets=[df[df['Google_Translation'].isna()].groupby(['date','sentiment']).aggregate({'date': 'count'})\
                .unstack().fillna(0).sum()['date',i] for i in ['positive','neutral','negative']]
    eng_tweets=[df[-(df['Google_Translation'].isna())].groupby(['date','sentiment']).aggregate({'date': 'count'})\
                .unstack().fillna(0).sum()['date',i] for i in ['positive','neutral','negative']]
    
    fig = make_subplots(rows=2, cols=2, column_widths=[0.7, 0.3],
                   specs=[[{"rowspan": 2,"type": "domain"},{"type": "domain"}],
                           [None,{"type": "domain"}]])
    fig.add_trace(go.Pie(labels=['Positive','Neutral','Negative'],values=all_tweets,marker_colors=['green','red','yellow'],title="All tweets"), row=1, col=1)
    fig.add_trace(go.Pie(labels=['Positive','Neutral','Negative'],values=som_tweets,title="Somali tweets"), row=1, col=2)
    fig.add_trace(go.Pie(labels=['Positive','Neutral','Negative'],values=eng_tweets,title='English tweets'), row=2, col=2)

    fig.update_layout(margin={"r": 0, "t": 0, "l": 10, "b": 0})
    return fig



def draw_sentiments(since, until, df, titre):
    d = df.groupby('date').aggregate({'date': 'count'})
    d_eng=df[df['Google_Translation'].isna()].groupby(['date','sentiment']).aggregate({'date': 'count'}).unstack()
    d_som=df[-(df['Google_Translation'].isna())].groupby(['date','sentiment']).aggregate({'date': 'count'}).unstack()
    # st.write(d)
    legend, tickes = echelle(since, until)
    X = [k for k in daterange(since, until)]
    x = [str(k) for k in daterange(since, until)]
    y = [d.loc[i].values[0] if i in d.index else 0 for i in x]
    y_eng_pos= [d_eng[('date', 'positive')].loc[i] if i in d_eng.index else 0 for i in x]
    y_eng_neg= [d_eng[('date', 'negative')].loc[i] if i in d_eng.index else 0 for i in x]
    y_eng_neu= [d_eng[('date', 'neutral')].loc[i] if i in d_eng.index else 0 for i in x]
    y_som_pos= [d_som[('date', 'positive')].loc[i] if i in d_som.index else 0 for i in x]
    y_som_neg= [d_som[('date', 'negative')].loc[i] if i in d_som.index else 0 for i in x]
    y_som_neu= [d_som[('date', 'neutral')].loc[i] if i in d_som.index else 0 for i in x]
    # st.write(y)
    fig = go.Figure(go.Scatter(name='Total week mean number', x=X, y=moyenne(y), mode='lines'))
    fig.add_trace(go.Bar(name='Positive in engligh', x=X, y=y_eng_pos,marker_color='darkgreen'))
    fig.add_trace(go.Bar(name='Positive in somali', x=X, y=y_som_pos,marker_color='forestgreen'))
    fig.add_trace(go.Bar(name='Neutral in english', x=X, y=y_eng_neu,marker_color='yellow'))
    fig.add_trace(go.Bar(name='Neutral in somali', x=X, y=y_som_neu,marker_color='khaki'))
    fig.add_trace(go.Bar(name='Negative in engligh', x=X, y=y_eng_neg,marker_color='red'))
    fig.add_trace(go.Bar(name='Negative in somali', x=X, y=y_som_neg,marker_color='coral'))
    fig.update_layout(barmode='stack')
    
    # fig.update_layout(autosize=True,legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="center",x=0.5))
    # fig.update_xaxes(ticktext=legend,tickvals=tickes)
    fig.update_layout(title=None,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=15),
                                  title=dict(font=dict(size=15))))
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    
    return fig


def waterfall(since,until,data,langage):
    df=data[(data.created_at.apply(lambda x: x.date())>=since.date())&\
            (data.created_at.apply(lambda x: x.date())<=until.date())].copy()
    if langage=='Somali':
        df=df[-(df['Google_Translation'].isna())]
    elif langage=='English':
        df=df[df['Google_Translation'].isna()]
    
    d=df.groupby(['date','sentiment']).aggregate({'date': 'count'}).unstack().fillna(0)
    a=d[('date', 'positive')]-d[('date', 'negative')]
    fig = go.Figure(go.Waterfall(
        orientation = "v",
        y = a.values,
        x = a.index,
        textposition = "outside",
        text = [str(i) for i in a.values],
    
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))

    fig.update_layout(
        title = None,
        showlegend = False
    )
    fig.update_layout(margin={"r": 0, "t": 10, "l": 10, "b": 10})
    return fig



def draw_numbers(since, until, df, titre):
    d = df.groupby('date').aggregate({'date': 'count'})
    d_eng=df[df['Google_Translation'].isna()].groupby('date').aggregate({'date': 'count'})
    d_som=df[-(df['Google_Translation'].isna())].groupby('date').aggregate({'date': 'count'})
    # st.write(d)
    legend, tickes = echelle(since, until)
    X = [k for k in daterange(since, until)]
    x = [str(k) for k in daterange(since, until)]
    y = [d.loc[i].values[0] if i in d.index else 0 for i in x]
    y_eng= [d_eng.loc[i].values[0] if i in d_eng.index else 0 for i in x]
    y_som= [d_som.loc[i].values[0] if i in d_som.index else 0 for i in x]
    # st.write(y)
    fig = go.Figure(go.Bar(name='Tweets in English per day', x=X, y=y_eng))
    fig.add_trace(go.Bar(name='Tweets in Somali per day', x=X, y=y_som))
    fig.update_layout(barmode='stack')
    fig.add_trace(go.Scatter(name='Total week mean number', x=X, y=moyenne(y), mode='lines'))

    # fig.update_layout(autosize=True,legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="center",x=0.5))
    # fig.update_xaxes(ticktext=legend,tickvals=tickes)
    fig.update_layout(title=None,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=15),
                                  title=dict(font=dict(size=15))))
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
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

