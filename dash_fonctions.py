import numpy as np
import plotly.graph_objects as go
import datetime

def draw_numbers(since, until, df, titre):
    d = df.groupby('date').aggregate({'date': 'count'})
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

