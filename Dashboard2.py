import dash
from dash import dcc
from dash import html
import plotly.graph_objs as go
import pandas as pd
from random import randint
from plotly.subplots import make_subplots

df = pd.read_csv('forDashboard.csv')

fig = go.Figure()

colors = {
    'background': '#52617D',
    'text': '#7FDBFF'
}

########################################################################################################################################################
def targetCountplot():

    fig = go.Figure()
    # groups
    good = df[df['Credit_Score']=='Good']
    standard = df[df['Credit_Score']=='Standard']
    poor = df[df['Credit_Score']=='Poor']


    fig.add_trace(go.Histogram(histfunc="count",x=good['Credit_Score'],name='Good'))
    fig.add_trace(go.Histogram(histfunc="count",x=standard['Credit_Score'],name='Standard'))
    fig.add_trace(go.Histogram(histfunc="count",x=poor['Credit_Score'],name='Poor'))

    fig.update_layout(xaxis_title='Credit Score', yaxis_title='Count',
                      plot_bgcolor=colors['background'],paper_bgcolor=colors['background'], font_color=colors['text'])

    return fig
########################################################################################################################################################
def missingValues():

    fig2 = go.Figure()

    x = df.isnull().mean() * 100

    #Creating a Color List
    colorList = []
    n = len(x)
    for i in range(n):
        colorList.append('#%06X' % randint(0, 0xFFFFFF))


    fig2 = go.Figure(go.Bar(x=df.columns, y=x,marker_color=colorList ))


    fig2.update_layout(xaxis_title='Feature', yaxis_title='%',
                      plot_bgcolor=colors['background'],paper_bgcolor=colors['background'], font_color=colors['text'],
                      title={
                           'text': "Missing Value Percentage",
                           'y': 0.95,
                           'x': 0.5,
                           'xanchor': 'center',
                           'yanchor': 'top',
                             }

                       )

    return fig2
########################################################################################################################################################
def featureHistogram():

    fig = make_subplots(rows=3, cols=2)

    fig.add_trace(
        go.Histogram(x=df['Monthly_Inhand_Salary'],name='Monthly Inhand Salary'),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=df['Delay_from_due_date'],name='Delay from Due Date'),
        row=1, col=2
    )
    fig.add_trace(
        go.Histogram(x=df['Credit_Utilization_Ratio'],name='Credit Utilization Ratio'),
        row=2, col=1
    )
    fig.add_trace(
        go.Histogram(x=df['Changed_Credit_Limit'],name='Changed Credit Limit'),
        row=2, col=2
    )
    fig.add_trace(
        go.Histogram(x=df['Outstanding_Debt'], name='Outstanding Debt'),
        row=3, col=1
    )
    fig.add_trace(
        go.Histogram(x=df['Changed_Credit_Limit'], name='Changed Credit Limit'),
        row=3, col=2
    )

    fig['layout']['xaxis']['title'] = 'Monthly Inhand Salary'
    fig['layout']['xaxis2']['title'] = 'Delay from Due Date'
    fig['layout']['xaxis3']['title'] = 'Credit Utilization Ratio'
    fig['layout']['xaxis4']['title'] = 'Changed Credit Limit'
    fig['layout']['xaxis5']['title'] = 'Outstanding Debt'
    fig['layout']['xaxis6']['title'] = 'Changed Credit Limit'

    fig.update_layout(showlegend=False,
                      plot_bgcolor=colors['background'],paper_bgcolor=colors['background'],
                      font_color=colors['text'],
                      title={
                          'text': "Histogram for Numerical Features",
                          'y': 0.95,
                          'x': 0.5,
                          'xanchor': 'center',
                          'yanchor': 'top'}
                      )
    return fig
########################################################################################################################################################




app = dash.Dash()


app.layout = html.Div(children=[

    html.H1(children='Target Variable Analysis',
            style={
                'textAlign': 'center',
                'color': 'black'
            }
            )
,

    html.Div(children=[
    html.H3(children='Count Plot for Target',
            style={
                'textAlign': 'left',
                'color': 'black'
            }
            ),

    dcc.Graph(
        id='credit-score',
        figure=targetCountplot())
]
),

html.Br(),
html.Br(),
html.Br(),
html.Br(),

html.H1(children='Multivariate Analysis',
            style={
                'textAlign': 'center',
                'color': 'black'
            }
            )
,
html.Div(children=[
    # html.H3(children='Missing Value %',
    #         style={
    #             'textAlign': 'left',
    #             'color': 'black'
    #         }
    #         )
    # html.Div(children='''Credit Score''',
    #          style={
    #              'textAlign': 'center',
    #              'color': 'blue'
    #          }
    #          ),
    dcc.Graph(
        id='graph2',
        figure=missingValues())
]
),

html.Br(),
html.Br(),

html.Div(children=[
    # html.H3(children='Histogram for Numerical Features',
    #         style={
    #             'textAlign': 'left',
    #             'color': 'black'
    #         }
    #         ),
    # html.Div(children='''Credit Score''',
    #          style={
    #              'textAlign': 'center',
    #              'color': 'blue'
    #          }
    #          ),
    dcc.Graph(
        id='graph3',
        figure=featureHistogram(),
        style={'height': '100%'})
]
        ,style={'height': '100vh'}
    # ,style={'height':'1000','width':'49%'}
)

    ]
    # ,style={'height': '100vh'}
)


app.run_server(debug=True)