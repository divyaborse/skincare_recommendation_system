
from dash import dash

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import base64
import pandas as pd
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt

import ast
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD

from scipy import sparse
from lightfm import LightFM
from lightfm.evaluation import auc_score
from lightfm.evaluation import precision_at_k, recall_at_k

import pickle

#image_filename = 'brushwall.png'
#encoded_image = base64.b64encode(open(image_filename, 'rb').read())

#image = Image.open('brushwall.png')
#backgroundImage = ImageTk.PhotoImage(image)
df = pd.read_csv('skindataall.csv', index_col=[0])

df1 = pd.read_csv('branddata.csv')
df2 = pd.read_csv('category.csv')

with open('mf_model.pkl', 'rb') as f:
    mf_model = pickle.load(f)


def dicts(df, colname):
    vals = list(set(df[colname]))
    l = []
    for i in vals:
        dic = {}
        dic['label'] = i
        dic['value'] = i
        l.append(dic)
    return l


tones_dict = dicts(df, 'Skin_Tone')
types_dict = dicts(df, 'Skin_Type')
eyes_dict = dicts(df, 'Eye_Color')
hair_dict = dicts(df, 'Hair_Color')

products_dictionary = dicts(df, 'Product')

brand_dictionary = dicts(df, 'Brand')

user_dictionary = dicts(df, 'User_id')
category_dictionary = dicts(df,'Category')


def Table(df):
    rows = []
    for i in range(len(df)):
        row = []
        for col in df.columns:
            value = df.iloc[i][col]
            # update this depending on which
            # columns you want to show links for
            # and what you want those links to be
            if col == 'Product':
                cell = html.Td(html.A(href=df.iloc[i]['Product_Url'], children=value))
            elif col == 'Product_Url':
                continue
            else:
                cell = html.Td(children=value)
            row.append(cell)
        rows.append(html.Tr(row))
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in ['Ratings', 'Products']])] + rows
    )


def Table1(df):
    rows = []
    for i in range(len(df)):
        row = []
        for col in df.columns:
            value = df.iloc[i][col]
            # update this depending on which
            # columns you want to show links for
            # and what you want those links to be
            if col == 'Product':
                cell = html.Td(html.A(href=df.iloc[i]['Product_Url'], children=value))
            elif col == 'Product_Url':
                continue
            else:
                cell = html.Td(children=value)
            row.append(cell)
        rows.append(html.Tr(row))
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in ['Product', 'Ingredients','Ratings']])] + rows
    )





def Table2(df):
    rows = []
    for i in range(len(df)):
        row = []
        for col in df.columns:
            value = df.iloc[i][col]
            # update this depending on which
            # columns you want to show links for
            # and what you want those links to be
            if col == 'Product':
                cell = html.Td(html.A(href=df.iloc[i]['Product_Url'], children=value))
            elif col == 'Product_Url':
                continue
            else:
                cell = html.Td(children=value)
            row.append(cell)
        rows.append(html.Tr(row))
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in ['Products','Brand']])] + rows
    )


def Table3(df):
    rows = []
    for i in range(len(df)):
        row = []
        for col in df.columns:
            value = df.iloc[i][col]
            # update this depending on which
            # columns you want to show links for
            # and what you want those links to be
            if col == 'Product':
                cell = html.Td(html.A(href=df.iloc[i]['Product_Url'], children=value))
            elif col == 'Product_Url':
                continue
            else:
                cell = html.Td(children=value)
            row.append(cell)
        rows.append(html.Tr(row))
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in ['Products','Category']])] + rows
    )


separation_string = '''
'''

intro_text = '''
Skin recommendations based on user features
'''

markdown_text_1 = '''
__Based on your features, these are the top products for you:__
'''

markdown_text_2 = '''
__Based on your preference, these are the top products for you:__
'''

markdown_text_3 = '''
__This user may like the following products:__
'''


def create_interaction_matrix(df, user_col, item_col, rating_col, norm=False, threshold=None):
    interactions = df.groupby([user_col, item_col])[rating_col].sum().unstack().reset_index().fillna(0).set_index(
        user_col)
    if norm:
        interactions = interactions.applymap(lambda x: 1 if x > threshold else 0)
    return interactions


interaction_matrix = create_interaction_matrix(df=df, user_col='User_id', item_col='Product_id',
                                               rating_col='Rating_Stars')


def create_user_dict(interactions):
    user_id = list(interactions.index)
    user_dict = {}
    counter = 0
    for i in user_id:
        user_dict[i] = counter
        counter += 1
    return user_dict


user_dict = create_user_dict(interaction_matrix)


def create_item_dict(df, id_col, name_col):
    item_dict = {}
    for i in df.index:
        item_dict[(df.loc[i, id_col])] = df.loc[i, name_col]
    return item_dict


product_dict = create_item_dict(df=df, id_col='Product_id', name_col='Product')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# https://codepen.io/chriddyp/pen/bWLwgP.css

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
     #'background': '#1DB954',
    "text": "#111111",
    "background-image" : "url('/assets/bg2.jpg')",
    "background-size": "cover",
}

app.layout = html.Div(style=colors, children=[
    html.H1(children='Skincare Recommendations',
            style={
                'textAlign': 'center',
                'color': colors['text'],
                'backgroundColor': colors["background-image"],
                'font-family': 'Bangers'
            }
            ),

    dcc.Markdown(children=intro_text),

    dcc.Markdown(children=separation_string),

    html.Label('Skin Tone'),
    dcc.Dropdown(
        id='skintone-selector',
        options=tones_dict,
        placeholder='Select your skin tone'
    ),

    html.Label('Skin Type'),
    dcc.Dropdown(
        id='skintype-selector',
        options=types_dict,
        placeholder='Select your skin type'
    ),

    html.Label('Eye color'),
    dcc.Dropdown(
        id='eyecolor-selector',
        options=eyes_dict,
        placeholder='Select your eye color'
    ),

    html.Label('Hair color'),
    dcc.Dropdown(
        id='haircolor-selector',
        options=hair_dict,
        placeholder='Select your eye color'
    ),

    dcc.Markdown(children=separation_string),

    dcc.Markdown(children=markdown_text_1),

    html.Div(id='output_1'),

    dcc.Markdown(children=separation_string),

    html.H2(children='Skincare recommendations based on your favorites',
            style={
                'textAlign': 'center',
                'color': colors['text'],
                'backgroundColor': colors["background-image"],
                'font-family': 'Bangers'
            }
            ),
    html.Label('Your favorite product!'),
    dcc.Dropdown(
        id='product-selector',
        options=products_dictionary,
        placeholder='Select your favorite product'
    ),

    dcc.Markdown(children=markdown_text_2),

    html.Div(id='output_2'),

    dcc.Markdown(children=separation_string),

    html.H2(children='Brand wise recommendation',
            style={
                'textAlign': 'center',
                'color': colors['text'],
                'backgroundColor': colors["background-image"],
                'font-family': 'Bangers'
            }
            ),
    html.Label('Brand product!'),
    dcc.Dropdown(
        id='brand-selector',
        options=brand_dictionary,
        placeholder='Select your favorite product'
    ),

    dcc.Markdown(children="Based on your preference, these are the top products for you:"),
    html.Div(id='output_3'),

    dcc.Markdown(children=separation_string),
html.H2(children='Product Categorization',
            style={
                'textAlign': 'center',
                'color': colors['text'],
                'backgroundColor': colors["background-image"],
                'font-family': 'Bangers'
            }
            ),
    html.Label('Category product!'),
    dcc.Dropdown(
        id='category-selector',
        options=category_dictionary,
        placeholder='Select product category'
    ),

    dcc.Markdown(children="Based on product category, these are the products for you:"),
    html.Div(id='output_4')


])


@app.callback(
    Output('output_1', 'children'),
    [Input('skintone-selector', 'value'),
     Input('skintype-selector', 'value'),
     Input('eyecolor-selector', 'value'),
     Input('haircolor-selector', 'value')])
def recommend_products_by_user_features(skintone, skintype, eyecolor, haircolor):
    ddf = df[(df['Skin_Tone'] == skintone) & (df['Hair_Color'] == haircolor) & (df['Skin_Type'] == skintype) & (
                df['Eye_Color'] == eyecolor)]
    if ddf.empty:
        return "Sorry no recommendation"
    else:

        recommendations = ddf[(ddf['Rating_Stars'].notnull())]
        data = recommendations[['Rating_Stars', 'Product_Url', 'Product']]
        data = data.sort_values('Rating_Stars', ascending=False).head()
        return Table(data)


@app.callback(
    Output('output_2', 'children'),
    [Input('product-selector', 'value')]
)
def content_recommender(product):
    try:
        df_cont = df[['Product', 'Product_id', 'Ingredients', 'Product_Url', 'Ing_Tfidf', 'Rating']]
        df_cont.drop_duplicates(inplace=True)#remove duplicates
        df_cont = df_cont.reset_index(drop=True)#reset the length of dataframes
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
        tfidf_matrix = tf.fit_transform(df_cont['Ingredients'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        titles = df_cont[['Product', 'Ing_Tfidf', 'Rating', 'Product_Url']]
        indices = pd.Series(df_cont.index, index=df_cont['Product'])
        idx = indices[product]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        product_indices = [i[0] for i in sim_scores]

    except KeyError:
        return None

    return Table1(titles.iloc[product_indices])


@app.callback(
    Output('output_3', 'children'),
    [Input('brand-selector', 'value')]
)
def brand_recommender(product):
    #df_dict = df1.loc[df1['Brand'] == product, 'Product'].tolist()
   # df_dict1 = df1.loc[df1['Brand'] == product, 'Product_Url'].to_dict()

    #return(Table2(df_dict))
    '''print('<table>')
    for sublist in df_dict:
        print('  <tr><td>')
        print
        ('    </td><td>'.join(sublist))
        print('  </td></tr>')
    print('</table>')'''

    cols = df1.columns[df1.columns.isin(['Product', 'Product_Url'])]
    df_dict = df1[(df1.Brand == product)]
    #print(df_dict)
    #return (html.Table([html.Tr([html.Td(html.A(col,href="https://plot.ly",target='_blank')) for col  in df_dict]),html.Br()]))
    '''for row in zip(df_dict):
        print(' '.join(row))'''
    #return ('\n'.join(map(str, df_dict)))
    #return df_dict
    return Table2(df_dict)



#product categorization
@app.callback(
    Output('output_4', 'children'),
    [Input('category-selector', 'value')]
)
def category_recommender(product):


    cols = df2.columns[df2.columns.isin(['Product', 'Product_Url'])]
    df_dict = df2[(df2.Category == product)]

    return Table3(df_dict)

if __name__ == '__main__':
    app.run_server(debug=True)
