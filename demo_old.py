import os
import io
import base64
import dash
import json
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import tensorflow as tf
import cv2
from io import BytesIO
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

external_script = ["https://tailwindcss.com/", {"src": "https://cdn.tailwindcss.com"}]

app = dash.Dash(__name__,  external_scripts=external_script,)


###BlackFigure

def blank_fig():
    fig = go.Figure(go.Scatter(x=[], y = []))
    fig.update_layout(template = None, width=10)
    fig.update_xaxes(showgrid = False, showticklabels = False, zeroline=False)
    fig.update_yaxes(showgrid = False, showticklabels = False, zeroline=False)
    
    return fig

###Layout


app.layout = html.Div(className="h-screen",
children=[
    html.Div(id="topbar", className="bg-[#F9D348] h-10"),
    html.Nav(id="navbar", className="max-w-7xl mx-auto px-2 sm:px-6 lg:px-10 mb-6",
    children=[
        html.Div(className="relative flex items-center justify-between",
        children=[
            html.Div(className="flex-1 flex items-center justify-center sm:items-stretch sm:justify-start",
            children=[
                html.Div(className="flex-shrink-0 flex items-center",
                children=[
                    html.Img(src= app.get_asset_url("SISR_Demologo.png"), className="block lg:hidden h-12 w-auto",),
                    html.Img(src= app.get_asset_url("SISR_Demologo.png"), className="hidden lg:block h-12 w-auto",)
                ]),
                html.Div(className="hidden sm:block sm:ml-6 pt-6",
                children=[
                    html.Div(className="flex space-x-6",
                    children=[
                        html.A(className="text-black-300 hover:underline decoration-4 hover:underline-offset-4 px-3 py-2 rounded-md text-base font-bold",children="Demo"),
                        html.A(className="text-black-300 hover:bg-gray-700 hover:text-white px-3 py-2 rounded-md text-base font-medium",children="Paper"),
                        html.A(className="text-black-300 hover:bg-gray-700 hover:text-white px-3 py-2 rounded-md text-base font-medium",children="About me"),
                    ])
                ])
            ])
        ])
    ]),

    html.Div(
        dcc.Upload(
        id='upload-image',
        children=html.Div(className="text-center text-black-200 mx-auto", children= [
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        className = "w-3/5 h-16 rounded-none border-2 border-black flex items-center mx-auto bg-[#FCF1E6] hover:translate-y-[4px] hover:-translate-x-[4px]  hover:drop-shadow-[8px_8px_0_rgba(0,0,0,1)] mb-6",
        # Allow multiple files to be uploaded
        multiple=True
        ),
        className="pt-5"
    ),
    
    html.Div(id='main',children=[
        html.Div(children=[
            html.Div(children=[
                dcc.Graph(id='upload', figure= blank_fig(), config={'displayModeBar': False, 'staticPlot': True}),
            ], className="items-center flex-1"),
            html.Div(children=[
                dcc.Graph(id='downgrade', figure= blank_fig()),
            ], className="items-center flex-1"),
            html.Div(children=[
                html.Ol("Way to use.",className="font-semibold text-xl list-decimal list-inside"),
                html.Li("Drag and Drop your picture."),
                html.Li(children=[
                    html.Li("Select scale:    ",),
                    dcc.Dropdown(['2','4'],'2', id='scale', className="w-20", clearable=False),
                ], className="flex items-center justify-center"), 
                html.Li("Select your area want to enhance."),
                html.Li("See the results."),
            ],className="flex-2 flex flex-col items-center justify-between h-80 pt-6"),

        ], className='flex flex-row items-start'),
        # html.H2("---"*25+"[ PREDICT ]"+"---"*25,className="pt-6"),
        html.Div(children=[
            html.Div(className="flex-auto border border-black"),
            html.Button("Predict", id="predict", className="w-24 h-10 text-center rounded-none border-2 border-black items-center bg-[#FCF1E6] hover:translate-y-[2px] hover:-translate-x-[2px]  hover:drop-shadow-[6px_6px_0_rgba(0,0,0,1)]  active:bg-gray-800 active:text-white"),
            html.Div(className="flex-auto border border-black"),
        ], className="flex flex-row items-center mb-6"),
        html.Div(children=[
            html.Div(children=[
                dcc.Graph(id='low_res', figure=  blank_fig(), config={'displayModeBar': False, 'staticPlot': True}),
            ], className="flex-1"),
            html.Div(children=[
                dcc.Graph(id='BICUBIC', figure=  blank_fig(), config={'displayModeBar': False, 'staticPlot': True}),
            ], className="flex-1"),
        ], className="flex items-center justify-evenly px-20"),

        html.Div(children=[
                html.Div(children=[

                    dcc.Graph(id='SRCNN', figure=  blank_fig(), config={'displayModeBar': False, 'staticPlot': True}),
                ], className="flex-1"),
                html.Div(children=[

                    dcc.Graph(id='EDSR', figure=  blank_fig(), config={'displayModeBar': False, 'staticPlot': True}),
                ], className="flex-1"),
            ], className="flex items-center justify-evenly px-20"),
    ],className="w-full px-10 flex-col items-center text-center")   
])


###Processing
def preprocess_b64(image_enc):
    """Preprocess b64 string into TF tensor"""
    decoded = base64.b64decode(str(image_enc).split("base64,")[-1])
    hr_image = tf.image.decode_image(decoded)

    if hr_image.shape[-1] == 4:
        hr_image = hr_image[..., :-1]
    # tf.expand_dims(tf.cast(hr_image, tf.float32), 0)
    return hr_image
    


def predict(lr, model):
    image = tf.expand_dims(lr, axis=0)
    image = tf.cast(image, tf.float32)
    sr=model(image)
    sr = model(image)
    sr = tf.clip_by_value(sr, 0, 255)
    sr = tf.round(sr)
    sr = tf.cast(sr, tf.uint8)
    sr = np.array(sr)
    sr = np.squeeze(sr,axis=0)
    return sr

def draw_plot(img, title):
    sr = px.imshow(img)
    sr.update_layout(title_font_family="inherit", title={
        'text': f"{title}",
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    return sr

def tf_to_b64(tensor, ext="jpeg"):
    buffer = BytesIO()
    image = tf.cast(tf.clip_by_value(tensor[0], 0, 255), tf.uint8).numpy()
    Image.fromarray(image).save(buffer, format=ext)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/{ext};base64,{encoded}"

def save_img(name,img):
    dash.no_update 
    if not os.path.exists("assets"):
        os.mkdir("assets")
    img = cv2.cvtColor(img.numpy(),cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"assets/{name}.jpeg",img)
    return f'assets/{name}.jpg'
###Model


def PSNR(super_resolution, high_resolution):
    """Compute the peak signal-to-noise ratio, measures quality of image."""
    # Max value of pixel is 255
    psnr_value = tf.image.psnr(high_resolution, super_resolution, max_val=255)[0]
    return psnr_value

##output
scl0 = None
scl = None
EDSR = None
SRCNN = None
BICUBIC = None

@app.callback(Output('upload', 'figure'),
           Output('downgrade', 'figure'),
            [Input('upload-image', 'contents')],
            [Input('scale', 'value')])
def update_output(content,scale):
    global scl0
    if content is None:
        return dash.no_update
    img = preprocess_b64(str(content))
    if scl0 ==  None  or  scl0  != scale:
        w,h,c = img.shape
        global down
        down = cv2.resize(img.numpy(),(h//int(scale),w//int(scale)),interpolation = cv2.INTER_CUBIC)
    global hr_img
    hr_img = img
    # lr = px.imshow(img)
    hr = draw_plot(img, "Low Resolution")
    return hr,draw_plot(down, "Low Resolution")

def get_model(scale):
    EDSR = tf.keras.models.load_model(f"assets/EDSR_x{scale}_l1.h5", custom_objects={"PSNR":PSNR})
    SRCNN = tf.keras.models.load_model(f"assets/SRCNN_x{scale}_l1.h5", custom_objects={"PSNR":PSNR})
    BICUBIC =  tf.keras.layers.UpSampling2D(size=int(scale), interpolation='bilinear')
    return EDSR, SRCNN, BICUBIC


@app.callback(
    Output('low_res', 'figure'),
    Output('BICUBIC', 'figure'),
    Output('SRCNN', 'figure'),
    Output('EDSR', 'figure'),
    [State('downgrade', 'relayoutData')],
    [State('scale', 'value')],
    [Input('predict', 'n_clicks')])
def display_relayout_data(relayoutData, scale, n_clicks):
    if 'xaxis.range[0]' in json.dumps(relayoutData):
        x_0 = int(relayoutData['xaxis.range[0]'])
        x_1 = int(relayoutData['xaxis.range[1]'])
        y_1 = int(relayoutData['yaxis.range[0]'])
        y_0 = int(relayoutData['yaxis.range[1]'])
        img = down[y_0:y_1,x_0:x_1]
        global hr_img
        # print("HR--",hr)
        hr_ = hr_img[(y_0*int(scale)):(y_1*int(scale)),(x_0*int(scale)):(x_1*int(scale))]
    
        global scl
        global SRCNN
        global EDSR
        global BICUBIC

        if scl ==  None  or  scl  != scale:
            scl = scale 
            EDSR, SRCNN, BICUBIC = get_model(scl)
            print(scale)
        edsr_img = predict(img,EDSR)
        srcnn_img = predict(img,SRCNN)
        bicubic_img = predict(img,BICUBIC)

        
        return draw_plot(hr_,"HR"), draw_plot(bicubic_img, "BICUBIC") ,draw_plot(srcnn_img, "SRCNN"), draw_plot(edsr_img, "EDSR")
    else:
        return dash.no_update

# 
app.run_server(debug=True, host='127.0.0.1')