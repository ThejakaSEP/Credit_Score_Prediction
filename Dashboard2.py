import pickle
import io
import base64
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
from random import randint
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
from dash import Dash, dcc, html, Input, Output, State

import numpy as np

df = pd.read_csv('forDashboard.csv')


def removeUnwanted(df):
    df = df[(df['Occupation'] != '_______')]
    df = df[(df['Credit_Mix'] != '_')]
    df = df[(df['Payment_Behaviour'] != '!@9#%8')]
    df = df[(df['Payment_of_Min_Amount'] != 'NM')]

    return df


df = removeUnwanted(df)

fig = go.Figure()

colors = {
    'background': '#52617D',
    'text': '#7FDBFF'
}


########################################################################################################################################################
def targetCountplot():
    fig = go.Figure()
    # groups
    good = df[df['Credit_Score'] == 'Good']
    standard = df[df['Credit_Score'] == 'Standard']
    poor = df[df['Credit_Score'] == 'Poor']

    fig.add_trace(go.Histogram(histfunc="count", x=good['Credit_Score'], name='Good'))
    fig.add_trace(go.Histogram(histfunc="count", x=standard['Credit_Score'], name='Standard'))
    fig.add_trace(go.Histogram(histfunc="count", x=poor['Credit_Score'], name='Poor'))

    fig.update_layout(xaxis_title='Credit Score', yaxis_title='Count',
                      plot_bgcolor=colors['background'], paper_bgcolor=colors['background'], font_color=colors['text'],
                      title={
                          'text': "Count Plot - Credit Score",
                          'y': 0.95,
                          'x': 0.5,
                          'xanchor': 'center',
                          'yanchor': 'top'}
                      )

    return fig


########################################################################################################################################################
def missingValues():
    fig2 = go.Figure()

    x = df.isnull().mean() * 100

    # Creating a Color List
    colorList = []
    n = len(x)
    for i in range(n):
        colorList.append('#%06X' % randint(0, 0xFFFFFF))

    fig2 = go.Figure(go.Bar(x=df.columns, y=x, marker_color=colorList))

    fig2.update_layout(xaxis_title='Feature', yaxis_title='%',
                       plot_bgcolor=colors['background'], paper_bgcolor=colors['background'], font_color=colors['text'],
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
        go.Histogram(x=df['Monthly_Inhand_Salary'], name='Monthly Inhand Salary'),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=df['Delay_from_due_date'], name='Delay from Due Date'),
        row=1, col=2
    )
    fig.add_trace(
        go.Histogram(x=df['Credit_Utilization_Ratio'], name='Credit Utilization Ratio'),
        row=2, col=1
    )
    fig.add_trace(
        go.Histogram(x=df['Changed_Credit_Limit'], name='Changed Credit Limit'),
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
                      plot_bgcolor=colors['background'], paper_bgcolor=colors['background'],
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
def heatmapGraph():
    fig2 = go.Figure()

    corr = df[df._get_numeric_data().columns].corr()

    fig2 = go.Figure(data=go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.columns)
    ))

    # fig2.update_layout(xaxis_title='Feature', yaxis_title='Feature',
    #                    plot_bgcolor=colors['background'], paper_bgcolor=colors['background'], font_color=colors['text'],
    #                    title={
    #                        'text': "Correlation Heat Map",
    #                        'y': 0.95,
    #                        'x': 0.5,
    #                        'xanchor': 'center',
    #                        'yanchor': 'top',
    #                    }
    #
    #                    )

    return fig2


########################################################################################################################################################
def densityPlot():
    fig2 = go.Figure()

    # corr = df[df._get_numeric_data().columns].corr()
    #
    fig2 = go.Figure(
        data=go.Pie(labels=df.groupby('Payment_Behaviour').size().index, values=df.groupby('Payment_Behaviour').size())
        )
    #
    fig2.update_layout(
        plot_bgcolor=colors['background'], paper_bgcolor=colors['background'], font_color=colors['text'],
        title={
            'text': "Credit Mix Composition",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        }

    )

    return fig2


########################################################################################################################################################


def classificationReportGraph():
    fig2 = go.Figure()

    X_test = pd.read_csv('X_test.csv')

    global y_test
    y_test = pd.read_csv('y_test.csv')

    global loaded_model
    loaded_model = pickle.load(open('RandomForestClassfier.sav', 'rb'))

    global predicted
    predicted = loaded_model.predict(X_test)

    precision, recall, fscore, support = score(y_test, predicted, average='macro')

    accuracyDict = {
        'Metrics': ['Precision', 'Recall', 'F1-Score'],
        'Values': [precision, recall, fscore]
    }

    accuracydf = pd.DataFrame(accuracyDict)

    fig2 = go.Figure([go.Bar(x=accuracydf['Metrics'], y=accuracydf['Values'])])

    fig2.update_layout(xaxis_title='Metric', yaxis_title='Value',
                       plot_bgcolor=colors['background'], paper_bgcolor=colors['background'], font_color=colors['text'],
                       title={
                           'text': "Model Metrices",
                           'y': 0.95,
                           'x': 0.5,
                           'xanchor': 'center',
                           'yanchor': 'top',
                       }

                       )

    return fig2


########################################################################################################################################################
def confusionMatrixGraph():
    fig2 = go.Figure()

    cm = confusion_matrix(y_test, predicted)

    cm_df = pd.DataFrame(cm,
                         index=['Good', 'Poor', 'Standard'],
                         columns=['Good', 'Poor', 'Standard'])

    fig2 = go.Figure(data=go.Heatmap(
        z=cm_df.values, x=cm_df.index, y=cm_df.columns
    ))

    fig2.update_layout(xaxis_title='Credit Score', yaxis_title='Credit Score',
                       plot_bgcolor=colors['background'], paper_bgcolor=colors['background'], font_color=colors['text'],
                       title={
                           'text': "Confusion Matrix",
                           'y': 0.95,
                           'x': 0.5,
                           'xanchor': 'center',
                           'yanchor': 'top',
                       }

                       )

    return fig2


########################################################################################################################################################
def predictor(df):
    # dfPredict = pd.read_csv(filePath)
    dfPredict = df
    prediction = loaded_model.predict(dfPredict)[0]

    classDict = {0: 'Credit Score Class : Good', 1: 'Credit Score Class: Poor', 2: 'Credit Score Class: Standard'}

    return classDict[prediction]


########################################################################################################################################################


app = dash.Dash()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div(children=[

    html.H1(children='Target Variable Analysis',
            style={
                'textAlign': 'center',
                'color': 'black'
            }
            )
    ,

    html.Div(children=[
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

        dcc.Graph(
            id='graph2',
            figure=missingValues())
    ]
    ),

    html.Br(),
    html.Br(),

    html.Div(children=[

        dcc.Graph(
            id='graph3',
            figure=featureHistogram(),
            style={'height': '100%'})
    ]
        , style={'height': '100vh'}
        # ,style={'height':'1000','width':'49%'}
    ),

    html.Br(),
    html.Br(),

    html.Div(children=[

        html.Div(

            dcc.Graph(
                id='graph4',
                figure=heatmapGraph(),
                style={'height': '100%'})
        ),
        html.Div(

            dcc.Graph(
                id='graph5',
                figure=densityPlot(),
                style={'height': '100%'})
        )
    ]
        , style={'height': '80vh', 'display': 'flex'}
    ),

    html.Br(),
    html.Br(),
    html.Br(),

    html.H1(children='Model Performances',

            style={
                'textAlign': 'center',
                'color': 'black'
            }
            ),

    html.Br(),

    html.H2(children='Random Forest Classifier',

            style={
                'textAlign': 'center',
                'color': 'black'
            }
            ),

    html.Div(children=[

        html.Div(

            dcc.Graph(
                id='graph6',
                figure=classificationReportGraph(),
                style={'height': '100%'})
        ),
        html.Div(

            dcc.Graph(
                id='graph7',
                figure=confusionMatrixGraph(),
                style={'height': '100%'})
        )
    ]
        , style={'height': '80vh', 'display': 'flex'}
    ),
    html.Br(),
    html.Br(),

    html.H1(children='Make Predictions',

            style={
                'textAlign': 'center',
                'color': 'black'
            }
            )
    ,
html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    )
]),
    # dcc.Upload(
    #     id='upload-data',
    #     children=html.Div([
    #         'Drag and Drop or ',
    #         html.A('Select Files')
    #     ]),
    #     style={
    #         'width': '100%',
    #         'height': '60px',
    #         'lineHeight': '60px',
    #         'borderWidth': '1px',
    #         'borderStyle': 'dashed',
    #         'borderRadius': '5px',
    #         'textAlign': 'center',
    #         'margin': '10px'
    #     },
    #     # Allow multiple files to be uploaded
    #     multiple=True
    # ),

    html.Div(id='output-data-upload', style={'color': 'blue', 'fontSize': 30, 'textAlign': 'center'}),

]
    # ,style={'height': '100vh'}
)


@app.callback(Output('output-data-upload', 'children'),
              [
                  Input('upload-data', 'contents'),
                  Input('upload-data', 'filename')
              ])
# ============================================================================================
def update_table(contents, filename):
    if contents:
        contents = contents[0]
        filename = filename[0]
        content_type, content_string = contents.split(',')

        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename:
                # Assume that the user uploaded a CSV or TXT file
                df = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')))
                print('Yes')
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])

        return predictor(df)


# ============================================================================================

app.run_server(debug=True)
# if __name__ == '__main__':
#     app.run_server(debug=False, host='0.0.0.0',port=8080)