# Import Modul Standar
import re
import joblib
import pandas as pd

# Import Modul Eksternal
import streamlit as st
from streamlit_option_menu import option_menu

# Atur layout halaman
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="💬",
    layout="wide",
)

# Sidebar untuk navigasi menggunakan streamlit-option-menu
with st.sidebar:
    selected = option_menu(
        "Analisis Sentimen",  # Judul Sidebar
        ["Home", "Tambah Data", "Klasifikasi", "Prediksi"],  # Menu
        icons=["house", "file-earmark-plus", "gear", "bar-chart"],  # Ikon
        menu_icon="menu-button",  
        default_index=0,  # Menu default yang dipilih
        styles={
            "container": {
                "padding": "10px 20px",  # Adjust padding for better spacing
                "background-color": "#2C3E50",  # Sidebar dark background
            },  
            "nav-link": {
                "font-size": "14px", 
                "margin": "8px 0",  # Vertical margin for better spacing
                "color": "#BDC3C7",  # Light text
                "padding": "8px 16px",  # Padding to make the links clickable and more spacious
                "border-radius": "5px",  # Optional: rounded corners for the links
            },  
            "nav-link-selected": {
                "background-color": "#1F2A35",  # Darker background for selected
                "color": "white",  # White text when selected
                "padding": "8px 16px",  # Maintain consistent padding
                "border-radius": "5px",  # Optional: rounded corners for the selected link
            },
            "menu-icon": {
                "color": "#BDC3C7",  # Light color for the menu icon
                "font-size": "20px",  # Slightly larger icon for better visibility
            }
        },
    )

# Fungsi header halaman
def header(title, subtitle):
    st.markdown(
        f"""
        <div style="background-color: #34495E; padding: 30px; border-radius: 5px; margin-bottom: 20px;">
            <h1 style="color: white; text-align: center; font-size: 30px;">{title}</h1>
            <p style="color: #BDC3C7; text-align: center; font-size: 18px;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

# Halaman Home
if selected == "Home":
    header("Analisis Sentimen Kelompok 8", "Final Project Mata Kuliah Data Mining")
    st.write("Anggota Kelompok:")
    st.write("1. Byanca Rebecca")
    st.write("2. Anisah Herian")
    st.write("3. Bintang Nuari")

# Halaman Tambah Data
elif selected == "Tambah Data":
    header("Tambah Data", "Unggah dataset untuk memulai analisis.")
    uploaded_file = st.file_uploader("Unggah file CSV", type="csv")
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            st.success("File berhasil diunggah!")
            st.write("Pratinjau Data:")
            st.dataframe(data.head())
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")

# Halaman Klasifikasi
elif selected == "Klasifikasi":
    header("Klasifikasi Data", "Evaluasi model untuk klasifikasi sentimen.")
    if st.session_state.data is None:
        st.warning("Silakan tambahkan data terlebih dahulu di halaman 'Tambah Data'.")
    else:
        st.write("Pratinjau Data:")
        st.dataframe(st.session_state.data.head())
        # Placeholder logika klasifikasi
        st.info("Klasifikasi akan diterapkan di sini.")

# Halaman Prediksi
elif selected == "Prediksi":
    header("Prediksi", "Gunakan model yang diunggah untuk analisis sentimen pada teks baru")

    # Unggah file model .pkl
    uploaded_model_file = st.file_uploader("Pilih file model (.pkl)", type="pkl")
    
    # Inisialisasi model
    model = None
    vectorizer = None  # Pastikan ini jika menggunakan vectorizer seperti TfidfVectorizer
    
    if uploaded_model_file is not None:
        try:
            model = joblib.load(uploaded_model_file)
            st.success("Model berhasil dimuat!")
            # Jika model membutuhkan vektorisasi, pastikan juga memuat vectorizer yang digunakan untuk pelatihan
            vectorizer_file = uploaded_model_file.name.replace('.pkl', '_vectorizer.pkl')
            try:
                vectorizer = joblib.load(vectorizer_file)
                st.success("Vectorizer berhasil dimuat!")
            except:
                st.warning("Vectorizer tidak ditemukan atau tidak digunakan.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memuat model: {e}")
    
    # Input teks untuk prediksi
    user_input = st.text_area("Masukkan teks untuk prediksi:")
    
    if st.button("Prediksi"):
        if model is None:
            st.warning("Harap unggah file model terlebih dahulu.")
        elif not user_input:
            st.warning("Harap masukkan teks untuk dianalisis.")
        else:
            try:
                # Preprocessing teks (misalnya, case folding)
                processed_input = user_input.lower()  # Sesuaikan dengan preprocessing yang diperlukan

                # Jika model menggunakan vektorisasi, transformasikan teks input
                if vectorizer:
                    processed_input = vectorizer.transform([processed_input])  # Vektorisasi teks

                # Lakukan prediksi menggunakan model
                prediction = model.predict(processed_input)
                st.success(f"Hasil Prediksi: {prediction[0]}")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

# Footer
st.markdown(
    """
    <div style="text-align: center; margin-top: 20px; font-size: small; color: #BDC3C7;">
        Dibuat dengan ❤️ menggunakan Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
