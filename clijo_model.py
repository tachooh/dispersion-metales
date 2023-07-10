import numpy as np
import pandas as pd
import math
import plotly.graph_objs as go
import streamlit as st

st.set_page_config(page_title = "Dispersión y fitorremdiación", page_icon = "", layout = "wide", initial_sidebar_state='collapsed')
st.header("Modelo de dispersión y fitorremediación de metales pesados en suelos")
option = st.selectbox('Selecciona la cantidad máxima de ángulos a graficar', ("", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

def make_grid(cols,rows):
    grid = [0]*cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows)
    return grid

def angulo_porcentaje(ang_p):
    total_porcentaje = 100

    for i in range(ang_p):
        grid = make_grid(2,(1,3,15,1))

        grid[0][0].markdown(f"<h5 style='text-align: right;'>{i + 1})</h5>", unsafe_allow_html=True)

        grid[0][1].markdown("<h5 style='text-align: center;'><br><br>Ángulo</h5>", unsafe_allow_html=True)
        angulo = grid[0][2].select_slider(f'{i + 1}. ', options = [f"{round((i/10 - 45), 1)}" for i in range(450*2 + 1)])
        angulos.append(float(str(angulo)))
        
        grid[1][1].markdown("<h5 style='text-align: center;'><br><br>Porcentaje</h5>", unsafe_allow_html=True)
        porcentaje = grid[1][2].select_slider(f'{i + 1}. ', options = [f"{i}%" for i in range(total_porcentaje + 1)])
        porcentajes.append((int(porcentaje.strip('%'))/100))

        st.write("---")

        total_porcentaje = int(total_porcentaje - porcentajes[i]*100)

        if total_porcentaje == 0.0:
            return 1

def convert_df(df):
    return df.to_csv(float_format="%.10g").encode('windows-1252')

def float_format(x, n):
    x = float(x)
    x = round(x, 3)
    x = str(x)
    x = x.replace(',', '')
    period_position = x.find(".")
    if period_position != -1:
        n_digits = len(x) - period_position - 1
        for i in range(n - n_digits):
            x += "0"
    x = x.replace('.', ',')

    return x

def desv_y(values):
    dsv = []
    for i in values:
        aux = 0
        # Estabilidad A
        a = 213
        aux = a*pow(i, 0.894)*A/100

        # Estabilidad B
        a = 156
        aux += a*pow(i, 0.894)*B/100

        # Estabilidad C
        a = 104
        aux += a*pow(i, 0.894)*C/100

        # Estabilidad D
        a = 68
        aux += a*pow(i, 0.894)*D/100

        # Estabilidad E
        a = 50.5
        aux += a*pow(i, 0.894)*E/100

        # Estabilidad F
        a = 34
        aux += a*pow(i, 0.894)*F/100
    
        dsv.append(aux)

    return dsv

def desv_z(values):
    dsv = []
    for i in values:
        aux = 0
        # Estabilidad A
        if i <= 1:
            c = 440.8
            d = 1.941
            f = 9.27
        else:
            c = 459.7
            d = 2.094
            f = -9.6

        aux = (c*pow(i, d) + f)*A/100

        # Estabilidad B
        if i <= 1:
            c = 106.6
            d = 1.149
            f = 3.3
        else:
            c = 108.2
            d = 1.098
            f = 2

        aux += (c*pow(i, d) + f)*B/100

        # Estabilidad C
        if i <= 1:
            c = 61
            d = 0.911
            f = 0
        else:
            c = 61
            d = 0.911
            f = 0

        aux += (c*pow(i, d) + f)*C/100

        # Estabilidad D
        if i <= 1:
            c = 33.2
            d = 0.725
            f = -1.7
        else:
            c = 44.5
            d = 0.516
            f = -13

        aux += (c*pow(i, d) + f)*D/100

        # Estabilidad E
        if i <= 1:
            c = 22.8
            d = 0.678
            f = -1.3
        else:
            c = 55.4
            d = 0.305
            f = -34

        aux += (c*pow(i, d) + f)*E/100

        # Estabilidad F
        if i <= 1:
            c = 14.35
            d = 0.74
            f = -0.35
        else:
            c = 62.6
            d = 0.18
            f = -48.6

        aux += (c*pow(i, d) + f)*F/100
        
        dsv.append(aux)

    return dsv

def concentracion(data, co):
    data_temp = data.iloc[:,:3]

    # Distancias en kilometros
    data_temp['dist_pos_x'] = np.sqrt((data_temp['x'] - 75)**2)/10 + 0.1
    
    # Desviaciones Y
    i = data_temp['dist_pos_x'].values
    data_temp['desv_y'] = pd.DataFrame({'desv_y': desv_y(i)})
    
    # Desviaciones Z
    i = data_temp['dist_pos_x'].values
    data_temp['desv_z'] = pd.DataFrame({'desv_z': desv_z(i)})
    
    # Concentraciones
    data_temp['concentracion'] = co*pow(math.e, -0.5*((data_temp['y']**2)/(data_temp['desv_y']**2) + (H**2)/(data_temp['desv_z']**2)))/(math.pi*data_temp['desv_y']*data_temp['desv_z']*vel_viento)
    C = periodo*24.5/(masa_molar*1000)*0.8
    data_temp['concentracion'] = data_temp['concentracion']*C

    # Ajustes para permitir graficar
    data_temp = data_temp.rename(columns = {'y' : 'x', 'x' : 'y'}).sort_values(by = ['y', 'x']).reset_index().drop(columns = 'index')
    data_temp['concentracion'] = data_temp['concentracion'].shift(75 - 1).fillna(0)
    data_temp.loc[data_temp['x'] < 75, 'concentracion'] = 0

    return data_temp['concentracion']

def adveccion(data):    
    data_temp = data.iloc[:,:3]

    Cr = v_darcy_limo / dm_l * 1 / 0.1
    shifted_data = data_temp.shift(1)
    shifted_data.fillna(0, inplace=True)
    mask = data_temp['y'] > 0
    data_temp.loc[mask, 'adv'] = (1 - Cr) * data_temp.loc[mask, 'z'] + Cr * shifted_data.loc[mask, 'z'] - con_ini
    
    return data_temp['adv']

def difusion(data):
    data_temp = data.iloc[:,:3]

    data_temp = data_temp.sort_values(by = ['x', 'y'], ascending = False)
    data_temp['dif_cont'] = data_temp.sort_values(by = ['x', 'y'], ascending = False)['z'].diff()
    data_temp = data_temp.sort_values(by = ['x', 'y'])
    data_temp.loc[data_temp['dif_cont'] < 0, 'dif_cont'] = 0
    data_temp['dif_cont'] = -data_temp['dif_cont']*dm/100
    
    return data_temp['dif_cont']

def fitorremediacion(data):
    data_temp = data.iloc[:,:3]

    data_temp['fito'] = data_temp['z']*masa_molar/22.4
    data_temp['fito'] = data_temp['fito']*multiplicador_etr
    data_temp['fito'] = data_temp['fito']*22.4/masa_molar
    data_temp['fito'] = data_temp['z'] - data_temp['fito']
    
    return -data_temp['fito']

# Información general 1
filas = 152
columnas = 152
Cantidad_celdas = int(filas*columnas)
num_celdas = filas*columnas
masa_molar = 64 #g/mol
H = 0
vel_viento = 2 # Metros/segundo

# Sidebar streamlit
st.sidebar.title('Determinación del periodo')
n_anos = st.sidebar.slider("Selecciona los años:", 1, 9, value = 1, step = 1)

st.sidebar.write('---')

st.sidebar.title('Concentración base')
con_base = st.sidebar.text_input("Selecciona la concentración base:")
con_ini = 0
if con_base:
    try:
        con_ini = float(con_base)
        st.sidebar.success(f"Ingresaste una concentración inicial de: {con_ini}")

    except ValueError:
        st.sidebar.warning("Concentración inválida, vuelve a ingresar un número.")
        con_ini = 0

st.sidebar.write('---')

total_value = 100
A = B = C = D = E = F = 0
st.sidebar.title("Estabilidad de Pasquill")

A = st.sidebar.select_slider('Selecciona el porcentaje de Estabilidad A', options = [f"{i}%" for i in range(total_value + 1)])
A = int(A.strip('%'))
total_value -= A

if (total_value != 0):
    B = st.sidebar.select_slider('Selecciona el porcentaje de Estabilidad B', options = [f"{i}%" for i in range(total_value + 1)])
    B = int(B.strip('%'))
    total_value -= B

    if (total_value != 0):
        C = st.sidebar.select_slider('Selecciona el porcentaje de Estabilidad C', options = [f"{i}%" for i in range(total_value + 1)])
        C = int(C.strip('%'))
        total_value -= C

        if (total_value != 0):
            D = st.sidebar.select_slider('Selecciona el porcentaje de Estabilidad D', options = [f"{i}%" for i in range(total_value + 1)])
            D = int(D.strip('%'))
            total_value -= D

            if (total_value != 0):
                E = st.sidebar.select_slider('Selecciona el porcentaje de Estabilidad E', options = [f"{i}%" for i in range(total_value + 1)])
                E = int(E.strip('%'))
                total_value -= E

                if (total_value != 0):
                    F = st.sidebar.select_slider('Selecciona el porcentaje de Estabilidad F', options = [f"{i}%" for i in range(total_value + 1)])
                    F = int(F.strip('%'))
                    total_value -= F

st.sidebar.write('---')

st.sidebar.title('Dispersión Gaussiana planta Enami Codelco')
con_g = st.sidebar.text_input("Selecciona el caudal Enami Codelco (g/s):")
co_gauss = 0
if con_g:
    try:
        co_gauss = float(con_g)
        st.sidebar.success(f"Ingresaste una concentración inicial de: {co_gauss}")

    except ValueError:
        st.sidebar.warning("Concentración inválida, vuelve a ingresar un número.")
        co_gauss = 0

pos_y = st.sidebar.slider("Selecciona una fila contaminada (y):", 1, 49, 25, step = 1)
if pos_y:
    pos_y += 50

pos_x = st.sidebar.slider("Selecciona una columna contaminada (x):", 1, 49, 25, step = 1)
if pos_x:
    pos_x += 50

st.sidebar.write('---')

st.sidebar.title('Dispersión Gaussiana planta Aesgener')
con_a = st.sidebar.text_input("Selecciona el caudal Aesgener (g/s):")
co_aesgener = 0
if con_a:
    try:
        co_aesgener = float(con_a)
        st.sidebar.success(f"Ingresaste una concentración inicial de: {co_aesgener}")

    except ValueError:
        st.sidebar.warning("Concentración inválida, vuelve a ingresar un número.")
        co_aesgener = 0

pos_y_aesgener = st.sidebar.slider("Selecciona una fila contaminada (y): ", 1, 49, 25, step = 1)
if pos_y_aesgener:
    pos_y_aesgener += 50

pos_x_aesgener = st.sidebar.slider("Selecciona una columna contaminada (x): ", 1, 49, 25, step = 1)
if pos_x_aesgener:
    pos_x_aesgener += 50

# Información general 2
periodo = n_anos*60*60*24*365 # Periodo en horas

# Datos Advección
dm_l = 0.27 # Porosidad eficaz del Limo / Silt
v_darcy_limo = 0.00001 # Kilometros / Segundos
dm = 0.52 # Porosidad eficaz

# Datos Fitorremediación
etp_anual = 0.15 # metro/año
temp_media_anual = 15.5 # T/año
etr_anual =  (etp_anual - math.pow(etp_anual,2)*(1/(0.8 + 0.14*temp_media_anual)))*(n_anos) # metro / año
multiplicador_etr = 1 - etr_anual

if (option != ""):
    angulos = []
    porcentajes = []
    angulo_porcentaje(int(option))

# Transformacion de tablas para graficar
    arreglo = np.full((filas, columnas), 0)
    df = pd.DataFrame(arreglo, columns=range(columnas))
    df['x'] = pd.DataFrame(np.tile(np.arange(filas), filas))
    df = pd.melt(df, id_vars=['x'], var_name='y', value_name='z')
    df['x'] = df['x'].apply(lambda x: int(x))
    df['y'] = df['y'].apply(lambda x: int(x))
    df = df[['y', 'x', 'z']]

    # Concentración de Gauss + Rotación
    df['co_gauss'] = concentracion(df, co_gauss)
    new = pd.DataFrame()
    i = 0
    for angulo in angulos:
        sin = math.sin(angulo*math.pi/180)
        cos = math.cos(angulo*math.pi/180)

        df['x_prev'] = df['x'] - 75
        df['y_prev'] = df['y'] - 75

        df['x_prev'] = round(df['x_prev']*cos - df['y_prev']*sin, 0).astype(int)
        df['y_prev'] = round(df['y_prev']*cos + df['x_prev']*sin, 0).astype(int)

        df['x_prev'] = df['x_prev'] + 75
        df['y_prev'] = df['y_prev'] + 75
        
    # Ajuste de variables para gráfico 50x50
        data = df.reset_index().drop(columns = 'index').drop_duplicates(['x_prev', 'y_prev'])

        data['y_prev'] -= 50 - (pos_y - 75)
        data['x_prev'] -= 50 - (pos_x - 75)

        data = data.loc[(data['x_prev'] >= 0) & (data['y_prev'] >= 0) & (data['x_prev'] <= 50) & (data['y_prev'] <= 50)]

    # Nuevo dataframe para graficar multiples angulos
        data = data.sort_values(by = ['y_prev','x_prev'])[['y_prev',
                                                'x_prev',
                                                'co_gauss',
                                            ]].reset_index().drop(columns = 'index')
        
        try:
            new['co_gauss'] += data['co_gauss']*porcentajes[i]
            
        except:
            new = data.copy()
            new['co_gauss'] *= porcentajes[i]
        i += 1

    df_gauss = new.copy()
    df = df[['y', 'x', 'z']]

    # Concentración de Aesgener + Rotación
    df['co_aesgener'] = concentracion(df, co_aesgener)
    new = pd.DataFrame()
    i = 0
    for angulo in angulos:
        sin = math.sin(angulo*math.pi/180)
        cos = math.cos(angulo*math.pi/180)

        df['x_prev'] = df['x'] - 75
        df['y_prev'] = df['y'] - 75

        df['x_prev'] = round(df['x_prev']*cos - df['y_prev']*sin, 0).astype(int)
        df['y_prev'] = round(df['y_prev']*cos + df['x_prev']*sin, 0).astype(int)

        df['x_prev'] = df['x_prev'] + 75
        df['y_prev'] = df['y_prev'] + 75
        
    # Ajuste de variables para gráfico 50x50
        data = df.reset_index().drop(columns = 'index').drop_duplicates(['x_prev', 'y_prev'])

        data['y_prev'] -= 50 - (pos_y_aesgener - 75)
        data['x_prev'] -= 50 - (pos_x_aesgener - 75)

        data = data.loc[(data['x_prev'] >= 0) & (data['y_prev'] >= 0) & (data['x_prev'] <= 50) & (data['y_prev'] <= 50)]

    # Nuevo dataframe para graficar multiples angulos
        data = data.sort_values(by = ['y_prev','x_prev'])[['y_prev',
                                                'x_prev',
                                                'co_aesgener',
                                            ]].reset_index().drop(columns = 'index')
        
        try:
            new['co_aesgener'] += data['co_aesgener']*porcentajes[i]
            
        except:
            new = data.copy()
            new['co_aesgener'] *= porcentajes[i]
        i += 1

    df_aesgener = new.copy()

    # Consolidación de concentraciones
    df = pd.merge(df_gauss, df_aesgener, how = 'inner', on = ['y_prev', 'x_prev']).rename(columns = {'y_prev' : 'y', 'x_prev' : 'x'})
    df['con_ini'] = con_ini
    df['z'] = df['co_gauss'] + df['co_aesgener'] + df['con_ini']
    df = df[['y', 'x', 'z', 'con_ini', 'co_gauss', 'co_aesgener']]

    df['adv'] = adveccion(df).fillna(0)
    df['z'] += df['adv']

    df['dif_cont'] = difusion(df).fillna(0)
    df['z'] += df['dif_cont']

    df['fito'] = fitorremediacion(df)
    df['z'] += df['fito']

    if (df['z'].sum()/len(df) != con_ini):
        df_plot = df[['x', 'y', 'z']].pivot(index='y', columns='x', values='z').reset_index().drop(columns = 'y')
        fig = go.Figure(data=[go.Surface(z=df_plot.values)])
        fig.update_layout(title='Distribución de concentración 3D (ppm)', autosize=False,
                    width=500, height=500,
                    margin = dict(l=65, r=50, b=65, t=90),
                    )
        st.plotly_chart(fig)

        detalle = df.copy()
        detalle = detalle.rename(columns = {'x' : 'Columna (x)',
                                            'y' : 'Fila (y)',
                                            'con_ini' : 'Concentración inicial (ppm)',
                                            'z' : 'Concentración final (ppm)',
                                            'co_gauss' : 'Concetración Gauss (ug/m^3)',
                                            'co_aesgener' : 'Concentración Aesgener (ug/m^3)',
                                            'adv' : 'Advección (ppm)',
                                            'dif_cont' : 'Difusión (ppm)',
                                            'fito' : 'Fitorremediación (ppm)'})
        
        detalle = detalle[['Fila (y)', 'Columna (x)', 'Concentración inicial (ppm)',
                                                      'Concentración final (ppm)',
                                                      'Concetración Gauss (ug/m^3)',
                                                      'Concentración Aesgener (ug/m^3)',
                                                      'Advección (ppm)',
                                                      'Difusión (ppm)',
                                                      'Fitorremediación (ppm)'
                                                      ]]

        st.dataframe(detalle)
        csv = convert_df(detalle)
        st.download_button(
        label="Descargar como CSV",
        data=csv,
        file_name='dispersion_metales.csv',
        mime='text/csv',)