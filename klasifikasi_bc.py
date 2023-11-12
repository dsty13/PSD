import streamlit as st
from pycaret.classification import *
from PIL import Image
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
st.title("Breast Cancer Prediction")
st.write('Merupakan aplikasi untuk memprediksi apakah kita memiliki kanker payudara Ganas atau kanker payudara Jinak')
selected = option_menu(
    menu_title=None,
    options=['Data','Implementasi','Profile'],
    orientation='horizontal',
    menu_icon=None,
    default_index=0,
    styles={
    "nav-link":{
        "font-size":"12px",
        "text-align":"center",
        "margin":"3px",
        "padding":"5px",
        "--hover-color":"#ff000000",},
        "nav-link-selected":{
        "background-color":"blue"},
    }
)
if selected == 'Data':
    st.title("Dataset Breast Cancer Wisconsin")
    st.write('Data yang digunakan yaitu dataset Breast Cancer Wisconsin yang saya peroleh dari UCI')
    st.write('Dataset Breast Cancer Wisconsin dapat diakses pada link : https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic')
    data = pd.read_csv('https://raw.githubusercontent.com/dsty13/dataset/main/breast-cancer.csv')
    data
    st.write('Tipe data yang yang digunakan pada dataset Breast Cancer Wisconsin adalah numerik dan kategorikal dengan jumlah 32 kolom dimana 30 kolom merupakan fitur, 1 kolom adalah ID dan 1 kolom merupakan label')
    st.write('Fitur dihitung dari gambar digital aspirasi jarum halus (FNA) dari massa payudara. Mereka menggambarkan karakteristik inti sel yang ada pada gambar. ')
    st.write('Berikut Merupakan Informasi Atribut yang ada dalam dataset Breast Cancer Wisconsin :')
    st.write('1. id: Nomor Id')
    st.write('2. diagnosis: M = malignant (ganas), B = benign (jinak)')
    st.write('3. radius_mean = Rata-rata jari-jari sel kanker (rata-rata jarak dari pusat ke titik-titik di sekelilingnya)')
    st.write('4. texture_mean = Rata-rata tekstur sel kanker(deviasi standar nilai skala abu-abu)')
    st.write('5. perimeter_mean = Rata-rata keliling sel kanker')
    st.write('6. area_mean = Rata-rata luas area sel kanker')
    st.write('7. smoothness_mean = Rata-rata kehalusan sel kanker(variasi lokal dalam panjang jari-jari)')
    st.write('8. compactness_mean = Rata-rata kompak sel kanker(keliling ^ 2 / luas - 1,0)')
    st.write('9. concavity_mean = Rata-rata cekungan sel kanker(tingkat keparahan bagian cekung dari kontur)')
    st.write('10. concave_points_mean = Rata-rata titik cekungan sel kanker(jumlah bagian cekung dari kontur)')
    st.write('11. symmetry_mean = Rata-rata simetri sel kanker')
    st.write('12. fractal_dimension_mean = Rata-rata dimensi fraktal sel kanker("coastline approximation" - 1)')
    st.write('13. radius_se =  Kesalahan standar dari jari-jari sel kanker')
    st.write('14. texture_se = Kesalahan standar dari tekstur sel kanker')
    st.write('15. perimeter_se = Kesalahan standar dari keliling sel kanker')
    st.write('16. area_se = Kesalahan standar dari luas area sel kanker')
    st.write('17. smoothness_se = Kesalahan standar dari kehalusan sel kanker')
    st.write('18. compactness_se = Kesalahan standar dari kompak sel kanker')
    st.write('19. concavity_se = Kesalahan standar dari cekungan sel kanker')
    st.write('20. concave_points_se = Kesalahan standar dari titik cekungan sel kanker')
    st.write('21. symmetry_se = Kesalahan standar dari simetri sel kanker')
    st.write('22. fractal_dimension_se = Kesalahan standar dari dimensi fraktal sel kanker')
    st.write('23. radius_worst =  Nilai terburuk dari jari-jari sel kanker')
    st.write('24. texture_worst = Nilai terburuk dari tekstur sel kanker')
    st.write('25. perimeter_worst = Nilai terburuk dari keliling sel kanker')
    st.write('26. area_worst = Nilai terburuk dari luas area sel kanker')
    st.write('27. smoothness_worst = Nilai terburuk dari kehalusan sel kanker')
    st.write('28. compactness_worst = Nilai terburuk dari kompak sel kanker')
    st.write('29. concavity_worst = Nilai terburuk dari cekungan sel kanker')
    st.write('30. concave points_worst = Nilai terburuk dari titik cekungan sel kanker')
    st.write('31. symmetry_worst = Nilai terburuk dari simetri sel kanker')
    st.write('32. fractal_dimension_worst = Nilai terburuk dari dimensi fraktal sel kanker')
    st.write('Dataset ini digunakan dalam analisis dan klasifikasi kanker payudara untuk memahami hubungan antara karakteristik sel kanker dengan tingkat keganasannya.')
    
if selected == 'Implementasi':
    st.title("Ayo Cek Apakah Anda memiliki kanker payudara Ganas atau kanker payudara Jinak")
    st.write('Isilah Form berikut untuk mengetahui apakah Anda memiliki kanker payudara Ganas atau kanker payudara Jinak')
    col1,col2=st.columns(2)
    with col1:
        radius_mean = st.number_input('Masukkan radius_mean :',format="%.7f")
        texture_mean = st.number_input('Masukkan texture_mean :',format="%.7f")
        perimeter_mean = st.number_input('Masukkan perimeter_mean :',format="%.7f")
        area_mean = st.number_input('Masukkan area_mean :',format="%.7f")
        smoothness_mean  = st.number_input('Masukkan smoothness_mean  :',format="%.7f")
        compactness_mean= st.number_input('Masukkan compactness_mean :',format="%.7f")
        concavity_mean = st.number_input('Masukkan concavity_mean :',format="%.7f")
        concave_points_mean = st.number_input('Masukkan concave_points_mean :',format="%.7f")
        symmetry_mean = st.number_input('Masukkan symmetry_mean :',format="%.7f")
        fractal_dimension_mean = st.number_input('Masukkan fractal_dimension_mean :',format="%.7f")
        radius_se = st.number_input('Masukkan radius_se:',format="%.7f")
        texture_se = st.number_input('Masukkan texture_se :',format="%.7f")
        perimeter_se = st.number_input('Masukkan perimeter_se :',format="%.7f")
        area_se = st.number_input('Masukkan area_se :',format="%.7f")
        smoothness_se = st.number_input('Masukkan smoothness_se :',format="%.7f")
    with col2:
        compactness_se = st.number_input('Masukkan compactness_se :',format="%.7f")
        concavity_se = st.number_input('Masukkan concavity_se :',format="%.7f")
        concave_points_se = st.number_input('Masukkan concave_points_se :',format="%.7f")
        symmetry_se = st.number_input('Masukkan symmetry_se :',format="%.7f")
        fractal_dimension_se = st.number_input('Masukkan fractal_dimension_se :',format="%.7f")
        radius_worst = st.number_input('Masukkan radius_worst :',format="%.7f")
        texture_worst = st.number_input('Masukkan texture_worst:',format="%.7f")
        perimeter_worst = st.number_input('Masukkan perimeter_worst :',format="%.7f")
        area_worst = st.number_input('Masukkan area_worst :',format="%.7f")
        smoothness_worst = st.number_input('Masukkan smoothness_worst :',format="%.7f")
        compactness_worst = st.number_input('Masukkan compactness_worst:',format="%.7f")
        concavity_worst = st.number_input('Masukkan concavity_worst :',format="%.7f")
        concave_points_worst = st.number_input('Masukkan concave_points_worst :',format="%.7f")
        symmetry_worst= st.number_input('Masukkan symmetry_worst :',format="%.7f")
        fractal_dimension_worst = st.number_input('Masukkan fractal_dimension_worst :',format="%.7f")
        
    button = st.button('Cek Prediksi Kanker Payudara',use_container_width = 500, type='primary')
    if button:
        if radius_mean !=0 and texture_mean !=0 and perimeter_mean !=0 and area_mean !=0 and smoothness_mean !=0 and compactness_mean!=0 and concavity_mean!=0 and concave_points_mean!=0 and symmetry_mean !=0 and fractal_dimension_mean !=0 and  radius_se!=0 and  texture_se!=0 and  perimeter_se!=0 and  area_se!=0 and  smoothness_se!=0 and compactness_se!=0 and  concavity_se!=0 and   concave_points_se!=0 and  symmetry_se!=0 and  fractal_dimension_se!=0 and  radius_worst!=0 and  texture_worst!=0 and  perimeter_worst!=0 and  area_worst!=0 and  smoothness_worst!=0 and  compactness_worst!=0 and  concavity_worst!=0 and  concave_points_worst!=0 and  symmetry_worst!=0 and  fractal_dimension_worst !=0 :
            fitur ={ "radius_mean":radius_mean,
                    "texture_mean":texture_mean,
                    "perimeter_mean":perimeter_mean,
                    "area_mean":area_mean,
                    "smoothness_mean":smoothness_mean,
                    "compactness_mean":compactness_mean,
                    "concavity_mean":concavity_mean,
                    "concave points_mean":concave_points_mean,
                    "symmetry_mean":symmetry_mean,
                    "fractal_dimension_mean":fractal_dimension_mean,
                    "radius_se":radius_se,
                    "texture_se":texture_se,
                    "perimeter_se":perimeter_se,
                    "area_se":area_se,
                    "smoothness_se":smoothness_se,
                    "compactness_se":compactness_se,
                    "concavity_se":concavity_se,
                    "concave points_se":concave_points_se,
                    "symmetry_se":symmetry_se,
                    "fractal_dimension_se":fractal_dimension_se,
                    "radius_worst":radius_worst,
                    "texture_worst":texture_worst,
                    "perimeter_worst":perimeter_worst,
                    "area_worst":area_worst,
                    "smoothness_worst":smoothness_worst,
                    "compactness_worst":compactness_worst,
                    "concavity_worst":concavity_worst,
                    "concave points_worst":concave_points_worst,
                    "symmetry_worst":symmetry_worst,
                    "fractal_dimension_worst":fractal_dimension_worst,
                
            }
            dt =pd.DataFrame(fitur,index=[0])
            import pickle
            loaded_model = load_model('lr_bc')
            prediction = predict_model(loaded_model,data=dt)
            prediksi=prediction['prediction_label']
            for i in prediksi:
                if i == 0:
                    st.write('Kanker Payudara Anda termasuk Jinak')
                else:
                    st.write('Kanker Payudara Anda termasuk Ganas')                       
        else:
            st.write('KOLOM BELUM TERISI')
            
if selected =='Profile':
    st.title("My Biodata")   
    st.write('Nama : Desti Fitrotun Nisa')
    st.write('Nim : 210411100182')
    st.write('Kelas : PSD A')