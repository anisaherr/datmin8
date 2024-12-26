# Import Modul Standar
import re
import joblib
import pandas as pd

# Import Modul Eksternal
import streamlit as st
from streamlit_option_menu import option_menu

# Fungsi untuk memuat model dan vectorizer dari file yang diupload
def load_model_and_vectorizer(model_file, vectorizer_file):
    try:
        model = joblib.load(model_file)
        vectorizer = joblib.load(vectorizer_file)
        return model, vectorizer
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model dan vectorizer: {e}")
        return None, None
        
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
        ["Home", "Tambah Data", "Pemodelan", "Prediksi"],  # Menu
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
    st.write("**Anggota Kelompok:**")
    st.write("1. Byanca Rebecca")
    st.write("2. Anisah Herian")
    st.write("3. Bintang Nuari")

# Halaman Tambah Data
elif selected == "Tambah Data":
    header("Tambah Data", "Unggah Dataset untuk Memulai Analisis")
    uploaded_file = st.file_uploader("Unggah file CSV", type="csv")
    if uploaded_file:
        try:
            # Membaca file CSV
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            st.success("File berhasil diunggah!")
            
            # Menampilkan pratinjau data
            st.write("Pratinjau Data:")
            st.dataframe(data.head())
            
            # Memastikan kolom Review Text dan Sentiment ada di data
            if 'Review Text' in data.columns and 'Sentiment' in data.columns:
                st.markdown("### Statistik Kata dan Sentimen")

                # Memisahkan data berdasarkan sentimen
                positive_reviews = data[data['Sentiment'] == 1]['Review Text']
                negative_reviews = data[data['Sentiment'] == 0]['Review Text']

                # Menggabungkan list kata menjadi satu string untuk analisis
                positive_text = " ".join([" ".join(eval(review)) if isinstance(review, str) else " ".join(review) for review in positive_reviews])
                negative_text = " ".join([" ".join(eval(review)) if isinstance(review, str) else " ".join(review) for review in negative_reviews])

                # Menghitung kata paling sering muncul
                from collections import Counter
                positive_words = Counter(positive_text.split()).most_common(15)
                negative_words = Counter(negative_text.split()).most_common(15)

                # Membuat tabel kata positif dan negatif
                combined_words = pd.DataFrame({
                    "Kata (Positif)": [word[0] for word in positive_words],
                    "Frekuensi (Positif)": [word[1] for word in positive_words],
                    "Kata (Negatif)": [word[0] for word in negative_words],
                    "Frekuensi (Negatif)": [word[1] for word in negative_words],
                })

                # Menampilkan tabel kata paling sering digunakan
                st.write("*15 Kata Paling Sering Muncul*")
                st.dataframe(combined_words)

                # Visualisasi jumlah sentimen
                st.markdown("### Visualisasi Jumlah Sentimen")
                sentiment_counts = data['Sentiment'].value_counts()
                sentiment_labels = ['Negatif', 'Positif']
                sentiment_values = [sentiment_counts.get(0, 0), sentiment_counts.get(1, 0)]

                # Plot bar chart
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 6))
                ax.bar(sentiment_labels, sentiment_values, color=['red', 'green'], alpha=0.8)
                ax.set_title("Jumlah Sentimen Positif dan Negatif", fontsize=16)
                ax.set_xlabel("Sentimen", fontsize=14)
                ax.set_ylabel("Jumlah", fontsize=14)
                for i, v in enumerate(sentiment_values):
                    ax.text(i, v + 2, str(v), ha='center', fontsize=12, color='black')
                st.pyplot(fig)

                # Membuat visualisasi Word Cloud
                st.markdown("### Visualisasi Word Cloud")
                from wordcloud import WordCloud

                wordcloud_positive = WordCloud(
                    width=500, height=500, background_color='white'
                ).generate(positive_text)

                wordcloud_negative = WordCloud(
                    width=500, height=500, background_color='white'
                ).generate(negative_text)

                # Menampilkan Word Cloud secara sejajar
                fig, ax = plt.subplots(1, 2, figsize=(15, 7))

                # Word Cloud Positif
                ax[0].imshow(wordcloud_positive, interpolation='bilinear')
                ax[0].set_title("Word Cloud Positif", fontsize=16)
                ax[0].axis('off')

                # Word Cloud Negatif
                ax[1].imshow(wordcloud_negative, interpolation='bilinear')
                ax[1].set_title("Word Cloud Negatif", fontsize=16)
                ax[1].axis('off')

                st.pyplot(fig)
            else:
                st.warning("Data harus memiliki kolom 'Review Text' dan 'Sentiment'.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")


# Halaman Pemodelan
elif selected == "Pemodelan":
    header("Pemodelan", "Evaluasi Model untuk Analisis Sentimen")
    
    # Memastikan st.session_state.data ada
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Silakan tambahkan data terlebih dahulu di halaman 'Tambah Data'.")
        st.stop()  # Menghentikan proses lebih lanjut jika data belum ada
    else:
        st.write("Pratinjau Data:")
        st.dataframe(st.session_state.data.head())

        # Memeriksa kolom 'Review Text' dan 'Sentiment' untuk melanjutkan
        if 'Review Text' not in st.session_state.data.columns or 'Sentiment' not in st.session_state.data.columns:
            st.error("Data harus memiliki kolom 'Review Text' dan 'Sentiment'.")
            st.stop()  # Menghentikan proses lebih lanjut jika kolom yang dibutuhkan tidak ada
        else:
            st.write("**Proses Vectorization (TF-IDF):**")
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
            from sklearn.svm import SVC
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.naive_bayes import MultinomialNB
            import matplotlib.pyplot as plt

            # TF-IDF Vectorization
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(st.session_state.data['Review Text'].astype(str))
            y = st.session_state.data['Sentiment']

            # Pilih model untuk dievaluasi
            st.write("**Pilih Model untuk Evaluasi:**")
            selected_models = st.multiselect(
                "Pilih model yang akan diuji",
                ["SVM", "KNN", "Random Forest", "Naive Bayes"],
                default=["SVM", "Random Forest"]
            )

            # Pilih proporsi data train:test
            st.write("**Pilih Proporsi Train:Test:**")
            test_sizes = st.multiselect(
                "Pilih proporsi data uji (%)",
                options=[10, 20, 30, 40, 50],
                default=[20]
            )

            # Tombol untuk memulai evaluasi
            if st.button("Mulai Evaluasi Model"):
                # Dictionary hasil evaluasi
                results = {
                    'Model': [],
                    'Split': [],
                    'Train Size': [],
                    'Test Size': [],
                    'Accuracy': [],
                    'Precision': [],
                    'Recall': [],
                    'F1-Score': []
                }

                # Inisialisasi model yang dipilih
                model_dict = {
                    "SVM": SVC(),
                    "KNN": KNeighborsClassifier(),
                    "Random Forest": RandomForestClassifier(random_state=42),
                    "Naive Bayes": MultinomialNB()
                }

                cm_figures = []  # List untuk menyimpan gambar Confusion Matrix
                fitted_models = {}  # Untuk menyimpan model yang sudah dilatih

                for model_name in selected_models:
                    model = model_dict[model_name]
                    for test_size in test_sizes:
                        split_ratio = f"{100 - test_size}:{test_size}"
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)

                        # Melatih model
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                        # Simpan model yang sudah dilatih
                        fitted_models[model_name] = model

                        # Menghitung metrik evaluasi
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                        # Simpan hasil evaluasi
                        results['Model'].append(model_name)
                        results['Split'].append(split_ratio)
                        results['Train Size'].append(X_train.shape[0])
                        results['Test Size'].append(X_test.shape[0])
                        results['Accuracy'].append(accuracy)
                        results['Precision'].append(precision)
                        results['Recall'].append(recall)
                        results['F1-Score'].append(f1)

                        # Simpan confusion matrix ke list dengan judul
                        cm = confusion_matrix(y_test, y_pred)
                        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negatif", "Positif"])
                        fig, ax = plt.subplots(figsize=(6, 5))
                        cm_display.plot(ax=ax, cmap=plt.cm.Blues)

                        # Menambahkan title dengan informasi model dan split
                        ax.set_title(f"{model_name} - Split {split_ratio}")

                        cm_figures.append(fig)

                # Menampilkan hasil evaluasi
                st.write("**Hasil Evaluasi Model:**")
                df_results = pd.DataFrame(results)
                st.dataframe(df_results)

                # Menampilkan semua Confusion Matrix dalam layout grid
                st.write("**Confusion Matrix untuk Semua Model:**")
                
                # Membuat layout grid untuk menampilkan confusion matrix dalam baris dan kolom
                num_columns = 2  # Jumlah kolom yang diinginkan (2 per baris)
                for i in range(0, len(cm_figures), num_columns):
                    cols = st.columns(num_columns)
                    for j in range(num_columns):
                        if i + j < len(cm_figures):  # Memastikan tidak melebihi jumlah gambar
                            with cols[j]:
                                st.pyplot(cm_figures[i + j])

                # Opsi untuk memilih model dan vectorizer untuk disimpan
                st.write("**Simpan Model dan Vectorizer:**")
                
                # Pilihan model yang ingin disimpan
                model_to_save = st.selectbox(
                    "Pilih model untuk disimpan",
                    options=selected_models,
                    index=0  # Default pilihan pertama
                )

                if st.button("Simpan Model dan Vectorizer"):
                    # Menyimpan model terpilih dan vectorizer
                    model_filename = f"{model_to_save}_model.pkl"
                    vectorizer_filename = "vectorizer.pkl"
                    
                    # Simpan model menggunakan joblib
                    joblib.dump(fitted_models[model_to_save], model_filename)
                    joblib.dump(vectorizer, vectorizer_filename)
                
                    # Pastikan file berhasil disimpan sebelum menyediakan tombol download
                    if os.path.exists(model_filename) and os.path.exists(vectorizer_filename):
                        # Menyediakan tombol untuk mengunduh file .pkl
                        with open(model_filename, "rb") as f:
                            st.download_button(
                                label=f"Unduh {model_to_save} Model (.pkl)",
                                data=f,
                                file_name=model_filename,
                                mime="application/octet-stream"
                            )
                
                        with open(vectorizer_filename, "rb") as f:
                            st.download_button(
                                label="Unduh Vectorizer (.pkl)",
                                data=f,
                                file_name=vectorizer_filename,
                                mime="application/octet-stream"
                            )
                    else:
                        st.error("Gagal menyimpan file model atau vectorizer.")

# Halaman Prediksi
elif selected == "Prediksi":
    header("Prediksi", "Gunakan Model untuk Analisis Sentimen")

    # Input untuk mengupload model dan vectorizer
    uploaded_model_file = st.file_uploader("Pilih file model (.pkl)", type="pkl")
    uploaded_vectorizer_file = st.file_uploader("Pilih file vectorizer (.pkl)", type="pkl")

    # Input teks untuk prediksi
    user_input = st.text_area("Masukkan Teks untuk Analisis:")

    if st.button("Prediksi"):
        if uploaded_model_file and uploaded_vectorizer_file:
            if user_input:
                # Memuat model dan vectorizer dari file yang diupload
                model, vectorizer = load_model_and_vectorizer(uploaded_model_file, uploaded_vectorizer_file)

                if model is not None and vectorizer is not None:
                    # Proses teks
                    processed_input = user_input.lower() 

                    # Vektorisasi teks, pastikan input dalam bentuk array 2D
                    input_tfidf = vectorizer.transform([processed_input])  # Input harus dalam bentuk list (2D array)

                    # Prediksi menggunakan model
                    prediction = model.predict(input_tfidf)

                    # Menentukan warna berdasarkan hasil prediksi
                    sentiment = 'Positif' if prediction[0] == 1 else 'Negatif'
                    color = 'green' if prediction[0] == 1 else 'red'

                    # Menampilkan hasil prediksi dengan warna
                    st.markdown(
                        f"""
                        <div style="text-align: center; font-size: 30px; font-weight: bold; color: {color};">
                            Hasil Sentimen: {sentiment}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.warning("Harap Masukkan Teks untuk Dianalisis!")
        else:
            st.warning("Harap Unggah File Model dan Vectorizer Terlebih Dahulu!")

# Footer
st.markdown(
    """
    <div style="text-align: center; margin-top: 20px; font-size: small; color: #BDC3C7;">
        Dibuat dengan ❤️ oleh Kelompok 8
    </div>
    """,
    unsafe_allow_html=True,
)
