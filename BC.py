import streamlit as st
from scipy.stats import kurtosis, skew
from pycaret.classification import *
from PIL import Image
from skimage import feature, color
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
    
    # Upload gambar dari user
    uploaded_image = st.file_uploader("Masukkan Gambar Breast Cancer...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image.")

        # Mengonversi gambar PIL ke array NumPy
        img_gray = np.array(Image.open(uploaded_image))

        # Konversi gambar ke citra grayscale jika diperlukan
        if len(img_gray.shape) == 3:  # Cek apakah citra berwarna
            img_gray = color.rgb2gray(img_gray)

        # Ekstraksi fitur menggunakan algoritma Canny Edge Detection
        edges = feature.canny(img_gray)
        
        # Menghitung fitur statistik untuk fitur-fitur yang diinginkan
        radius_mean = np.mean(img_gray)
        texture_mean = np.std(img_gray)
        perimeter_mean = np.sum(edges)
        area_mean = np.sum(img_gray)
        smoothness_mean = np.var(img_gray)
        compactness_mean = (perimeter_mean**2) / area_mean - 1.0
        concavity_mean = np.sum(edges)
        concave_points_mean = np.sum(edges)
        symmetry_mean = np.mean(img_gray)
        fractal_dimension_mean = np.var(img_gray) - 1
        
        radius_se = np.std(img_gray)
        texture_se = np.mean(img_gray)
        perimeter_se = np.sum(edges)
        area_se = np.sum(img_gray)
        smoothness_se = np.var(img_gray)
        compactness_se = (perimeter_se**2) / area_se - 1.0
        concavity_se = np.sum(edges)
        concave_points_se = np.sum(edges)
        symmetry_se = np.mean(img_gray)
        fractal_dimension_se = np.var(img_gray) - 1

        radius_worst = np.max(img_gray)
        texture_worst = np.mean(img_gray)
        perimeter_worst = np.sum(edges)
        area_worst = np.sum(img_gray)
        smoothness_worst = np.var(img_gray)
        compactness_worst = (perimeter_worst**2) / area_worst - 1.0
        concavity_worst = np.sum(edges)
        concave_points_worst = np.sum(edges)
        symmetry_worst = np.mean(img_gray)
        fractal_dimension_worst = np.var(img_gray) - 1
        
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
        #st.write(dt)
        import pickle
        with open('scaler_BC.pkl','rb') as prepro:
            skala = pickle.load(prepro)
        data_norm = skala.transform(dt) 
        #st.write('---------- Data hasil Normalisasi ----------')
        #st.write(data_norm)
        with open('PCA_BC.pkl','rb') as pca_bc:
            pca =pickle.load(pca_bc)
        data_pca = pca.transform(data_norm)
        #st.write('---------- Data hasil Ekstraksi Fitur dengan PCA ----------')
        #st.write(data_pca)
        with open('BC_lr.pkl','rb') as pcaknn :
            knn_pca = pickle.load(pcaknn)
        predict_pca = knn_pca.predict(data_pca)
        for i in predict_pca:
            if i == 1:
                st.write('Kanker Payudara Anda termasuk Ganas')
            else:
                st.write('Kanker Payudara Anda termasuk Jinak')
        #loaded_model = load_model('lr_BC')
        #prediction = predict_model(loaded_model,data=dt)
        #prediksi=prediction['prediction_label']
        #for i in prediksi:
            #if i == 0:
                #st.write('Kanker Payudara Anda termasuk Jinak')
            #else:
                #st.write('Kanker Payudara Anda termasuk Ganas')

            
if selected =='Profile':
    st.title("My Biodata")   
    st.write('Nama : Desti Fitrotun Nisa')
    st.write('Nim : 210411100182')
    st.write('Kelas : PSD A')