# Import Modul Standar
import re
import joblib
import pandas as pd

# Import Modul Eksternal
import streamlit as st
from streamlit_option_menu import option_menu
from nltk.tokenize import word_tokenize
from transformers import pipeline

# Import Modul Pustaka Bahasa Indonesia
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Inisialisasi model analisis sentimen
sentiment_analyzer = pipeline("sentiment-analysis")

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
        ["Home", "Tambah Data", "Preprocessing", "Klasifikasi", "Prediksi", "Visualisasi"],  # Menu
        icons=["house", "file-earmark-plus", "gear", "bar-chart", "search", "graph-up-arrow"],  # Ikon
        menu_icon="menu-button",  # Ikon utama sidebar
        default_index=0,  # Menu default yang dipilih
        styles={
            "container": {"padding": "0!important", "background-color": "#2C3E50"},  # Sidebar dark background
            "nav-link": {"font-size": "14px", "margin": "5px", "color": "#BDC3C7"},  # Light text
            "nav-link-selected": {"background-color": "#1F2A35", "color": "white"},  # Darker background for selected
        },
    )

# Fungsi header halaman
def header(title, subtitle):
    st.markdown(
        f"""
        <div style="background-color: #34495E; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
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

# Halaman Preprocessing
elif selected == "Preprocessing":
    header("Preprocessing Data", "Bersihkan dan siapkan data untuk analisis.")
    if st.session_state.data is None:
        st.warning("Silakan tambahkan data terlebih dahulu di halaman 'Tambah Data'.")
    else:
        st.write("Pratinjau Data:")
        st.dataframe(st.session_state.data.head())

        # Preprocessing contoh
        st.write("**Proses Preprocessing:**")
        data = st.session_state.data.copy()
        data['Review Text'] = data['Review Text'].str.lower()
        st.write("1. Case Folding selesai.")

        # Hapus URL
        data['Review Text'] = data['Review Text'].apply(lambda x: re.sub(r'https?://\S+|www\.\S+', '', str(x)))
        st.write("2. Penghapusan URL selesai.")

        # Hapus Tanda Baca
        data['Review Text'] = data['Review Text'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))
        st.write("3. Penghapusan tanda baca selesai.")

        # Hapus Angka
        data['Review Text'] = data['Review Text'].apply(lambda x: re.sub(r'\d+', '', str(x)))
        st.write("4. Penghapusan angka selesai.")

        # Hapus Emoji
        emoji_pattern = re.compile(
        "[" 
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+", flags=re.UNICODE)
        data['Review Text'] = data['Review Text'].apply(lambda x: emoji_pattern.sub(r'', str(x)))
        st.write("5. Penghapusan emoji selesai.")

        # Tokenisasi
        data['Tokens'] = data['Review Text'].apply(word_tokenize)
        st.write("6. Tokenisasi selesai.")

        # Stopword Removal
        stopword_factory = StopWordRemoverFactory()
        stopword_remover = stopword_factory.create_stop_word_remover()
        data['Tokens'] = data['Tokens'].apply(
                    lambda x: [word for word in x if word not in stopword_factory.get_stop_words()]
                )
        st.write("7. Penghapusan stopword selesai.")

        # Stemming
        stemmer_factory = StemmerFactory()
        stemmer = stemmer_factory.create_stemmer()
        data['Tokens'] = data['Tokens'].apply(lambda x: [stemmer.stem(word) for word in x])
        st.write("8. Stemming selesai.")

        # POS Tagging
        data['POS Tagged'] = data['Tokens'].apply(pos_tag)
        st.write("9. POS Tagging selesai.")

        # Tampilkan hasil
        st.write("Hasil Preprocessing dan POS Tagging:")
        st.dataframe(data.head())

        # Simpan hasil dan beri opsi unduh
        processed_file = 'processed_data.csv'
        data.to_csv(processed_file, index=False)
        st.download_button(
            label="Unduh File Preprocessing",
            data=open(processed_file, "rb"),
            file_name="processed_data.csv",
            mime="text/csv"
        )
        st.dataframe(data.head())
        st.session_state.data = data

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

# Halaman Visualisasi
elif selected == "Visualisasi":
    header("Visualisasi Data", "Lihat distribusi data sentimen.")
    if st.session_state.data is None:
        st.warning("Silakan tambahkan data terlebih dahulu di halaman 'Tambah Data'.")
    else:
        st.write("Pratinjau Data:")
        st.dataframe(st.session_state.data.head())

        # Contoh visualisasi
        st.write("Distribusi Sentimen:")
        data = st.session_state.data
        if "Sentiment" in data.columns:
            sentiment_counts = data["Sentiment"].value_counts()
            st.bar_chart(sentiment_counts)
        else:
            st.warning("Kolom 'Sentiment' tidak ditemukan dalam data.")

# Footer
st.markdown(
    """
    <div style="text-align: center; margin-top: 20px; font-size: small; color: #BDC3C7;">
        Dibuat dengan ❤️ menggunakan Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
