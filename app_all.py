import os

# Включаем ускорение Xet
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = "50"
os.environ["HF_XET_MAX_CONCURRENT_DOWNLOADS"] = "8"

import streamlit as st
import torch
import librosa
import json
import tempfile
import io
import soundfile as sf
import numpy as np
import ast
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
import pandas as pd
import re
import requests
from nltk.stem import SnowballStemmer
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="ТНВЭД Голосовой Ассистент",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 Голосовой помощник для определения кода ТНВЭД")
st.markdown("""
**Описание проекта:**  
Этот голосовой ассистент помогает определить код ТНВЭД для товара по голосовому описанию.  
Просто скажите или загрузите аудиофайл с описанием товара, и модель распознает речь и подберет код.
""")

# ===== ЗАГРУЗКА БАЗЫ ДАННЫХ ИЗ GOOGLE DRIVE =====
@st.cache_resource
def load_database():
    """Загружает базу данных ТНВЭД из Google Drive"""
    try:
        # Ссылка для скачивания
        url = "https://drive.google.com/u/0/uc?id=149QYvV0RFbj0NQ3W1nW4naq_vcRuwnEB&export=download"
        
        with st.spinner("📥 Загрузка базы данных ТНВЭД из Google Drive..."):
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Сохраняем во временный файл
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                tmp_file.write(response.content)
                excel_path = tmp_file.name
            
            # Читаем Excel
            df = pd.read_excel(excel_path)
            st.success(f"✅ Загружено {len(df)} товаров из базы данных")
            return df
    except Exception as e:
        st.error(f"❌ Ошибка загрузки базы данных: {e}")
        return None

# ===== ИНСТРУМЕНТ ПОИСКА =====
class SmartSearchTool:
    def __init__(self, df):
        """Инициализация с DataFrame из Excel"""
        self.df = df
        self.stemmer = SnowballStemmer("russian")
        self._build_index()
    
    def _stem_text(self, text):
        words = re.findall(r'\b[а-яa-z]{2,}\b', text.lower())
        return [self.stemmer.stem(word) for word in words]
    
    def _get_ngrams(self, text, n=3):
        text = re.sub(r'[^а-яa-z]', '', text.lower())
        return [text[i:i+n] for i in range(len(text)-n+1)]
    
    def _build_index(self):
        self.word_index = {}
        self.ngram_index = {}
        
        # Определяем колонку с названием товара
        product_col = 'product_name' if 'product_name' in self.df.columns else 'product'
        
        for idx, row in self.df.iterrows():
            name = str(row[product_col]).lower()
            stems = self._stem_text(name)
            for stem in set(stems):
                if stem not in self.word_index:
                    self.word_index[stem] = []
                self.word_index[stem].append(idx)
            ngrams = self._get_ngrams(name, n=3)
            for ngram in set(ngrams):
                if ngram not in self.ngram_index:
                    self.ngram_index[ngram] = []
                self.ngram_index[ngram].append(idx)
    
    def search(self, query, top_k=5):
        query = str(query).lower()
        results = {}
        
        # Поиск по стеммингу
        query_stems = self._stem_text(query)
        for stem in query_stems:
            for idx in self.word_index.get(stem, []):
                results[idx] = results.get(idx, 0) + 2
        
        # Поиск по n-граммам
        query_ngrams = self._get_ngrams(query, n=3)
        for ngram in query_ngrams:
            for idx in self.ngram_index.get(ngram, []):
                results[idx] = results.get(idx, 0) + 1
        
        # Точное совпадение слов
        query_words = set(re.findall(r'\b[а-яa-z]{2,}\b', query))
        product_col = 'product_name' if 'product_name' in self.df.columns else 'product'
        for idx, row in self.df.iterrows():
            if idx in results:
                name_words = set(re.findall(r'\b[а-яa-z]{2,}\b', str(row[product_col]).lower()))
                common = query_words & name_words
                if common:
                    results[idx] += len(common) * 3
        
        final_results = []
        for idx, score in sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            row = self.df.iloc[idx]
            product_col = 'product_name' if 'product_name' in self.df.columns else 'product'
            hs_col = 'hs_code' if 'hs_code' in self.df.columns else 'code'
            final_results.append({
                'product_name': row[product_col],
                'hs_code': row[hs_col],
                'score': score
            })
        
        return final_results

# ===== ЗАГРУЗКА МОДЕЛИ =====
@st.cache_resource
def load_model():
    with st.spinner("📥 Загрузка модели Gemma 3n на CPU... Это может занять 5-10 минут..."):
        HF_TOKEN = "hf_token"
        
        try:
            processor = AutoProcessor.from_pretrained(
                "google/gemma-3n-e2b-it",
                token=HF_TOKEN
            )
            
            model = Gemma3nForConditionalGeneration.from_pretrained(
                "google/gemma-3n-e2b-it",
                token=HF_TOKEN,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True
            ).eval()
            
            st.success("✅ Модель загружена на CPU!")
            return model, processor
        except Exception as e:
            st.error(f"❌ Ошибка загрузки: {e}")
            return None, None

# ===== ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ =====
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "model" not in st.session_state:
    st.session_state.model = None
if "processor" not in st.session_state:
    st.session_state.processor = None
if "result" not in st.session_state:
    st.session_state.result = None
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "searcher" not in st.session_state:
    st.session_state.searcher = None
if "db_loaded" not in st.session_state:
    st.session_state.db_loaded = False

# ===== ЗАГРУЗКА БАЗЫ ДАННЫХ =====
if not st.session_state.db_loaded:
    df = load_database()
    if df is not None:
        st.session_state.searcher = SmartSearchTool(df)
        st.session_state.db_loaded = True

# ===== ЗАГРУЗКА МОДЕЛИ =====
if not st.session_state.model_loaded:
    model, processor = load_model()
    if model is not None:
        st.session_state.model = model
        st.session_state.processor = processor
        st.session_state.model_loaded = True

# ===== ФУНКЦИЯ ОБРАБОТКИ АУДИО С ИНСТРУМЕНТОМ =====
def process_audio_file(audio_path):
    if st.session_state.searcher is None:
        return {"error": "База данных не загружена"}, None
    
    try:
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path, "sampling_rate": sr},
                    {
                        "type": "text",
                        "text": "Распознай товар в аудио и верни JSON: {'product': 'название товара'}. БЕЗ пояснений."
                    }
                ]
            }
        ]
        
        inputs = st.session_state.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(st.session_state.model.device)
        
        with torch.no_grad():
            output_ids = st.session_state.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False
            )
        
        generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
        response_text = st.session_state.processor.decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        # Извлекаем товар из ответа
        product = None
        try:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = response_text[start:end]
                result = ast.literal_eval(json_str)
                product = result.get('product')
        except:
            pass
        
        if not product:
            return {"error": "Не удалось распознать товар", "raw": response_text}, None
        
        # Ищем код в базе данных через инструмент
        search_results = st.session_state.searcher.search(product)
        
        if search_results:
            best = search_results[0]
            return {
                "product": product,
                "hs_code": best['hs_code'],
                "found_in_excel": True,
                "all_matches": search_results
            }, None
        else:
            return {
                "product": product,
                "hs_code": None,
                "found_in_excel": False
            }, None
            
    except Exception as e:
        return None, str(e)

# ===== ВКЛАДКИ =====
tab1, tab2 = st.tabs(["🎤 Запись с микрофона", "📂 Загрузка аудиофайла"])

with tab1:
    st.subheader("🎤 Скажите описание товара")
    
    try:
        from streamlit_mic_recorder import mic_recorder
        audio = mic_recorder(
            start_prompt="🎤 Нажмите и скажите",
            stop_prompt="⏹️ Остановить запись",
            just_once=True,
            use_container_width=True,
            key="mic_recorder"
        )
    except ImportError:
        st.error("⚠️ Библиотека streamlit-mic-recorder не установлена.")
        audio = None

    if audio is not None and isinstance(audio, dict) and 'bytes' in audio:
        audio_bytes = audio['bytes']
        
        try:
            import av
            
            input_buffer = io.BytesIO(audio_bytes)
            container = av.open(input_buffer)
            stream = container.streams.audio[0]
            native_sr = stream.rate
            
            audio_frames = []
            for frame in container.decode(stream):
                array = frame.to_ndarray()
                if array.dtype.kind in ['i', 'u']:
                    array = array.astype(np.float32) / np.iinfo(array.dtype).max
                else:
                    array = array.astype(np.float32)
                if array.ndim > 1:
                    array = np.mean(array, axis=0)
                else:
                    array = array.flatten()
                audio_frames.append(array)
            
            if audio_frames:
                full_audio_data = np.concatenate(audio_frames)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    sf.write(tmp_file.name, full_audio_data, samplerate=native_sr, format='WAV', subtype='PCM_16')
                    st.session_state.audio_path = tmp_file.name
                st.success(f"✅ Аудио записано! Размер: {len(audio_bytes) / 1024:.1f} KB")
            else:
                st.error("❌ Запись пустая")
                
        except Exception as e:
            st.error(f"❌ Ошибка: {e}")
            st.session_state.audio_path = None
        
        st.audio(audio_bytes, format='audio/wav')
        st.session_state.result = None

    if st.session_state.audio_path is not None and os.path.exists(st.session_state.audio_path):
        st.write("---")
        st.write("**📁 Аудио сохранено**")
        
        if st.button("🔍 Определить код ТНВЭД", type="primary"):
            if not st.session_state.model_loaded:
                st.error("⚠️ Модель не загружена.")
            elif st.session_state.searcher is None:
                st.error("⚠️ База данных не загружена.")
            else:
                with st.spinner("🎧 Обработка аудио..."):
                    result, error = process_audio_file(st.session_state.audio_path)
                    if error:
                        st.error(f"❌ Ошибка: {error}")
                    else:
                        st.session_state.result = result
                        st.success("✅ Готово! Результат внизу.")
    else:
        st.info("⏳ Запишите аудио с помощью микрофона.")

with tab2:
    st.subheader("📂 Загрузите готовый аудиофайл")
    
    uploaded_file = st.file_uploader(
        "Выберите аудиофайл",
        type=["wav", "mp3", "flac", "m4a"],
        help="Поддерживаются форматы: WAV, MP3, FLAC, M4A"
    )
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            audio_path = tmp_file.name
        
        st.info(f"📁 Имя: {uploaded_file.name}")
        st.info(f"📊 Размер: {uploaded_file.size / 1024:.1f} KB")
        
        if st.button("🔍 Определить код ТНВЭД (файл)", type="primary"):
            if not st.session_state.model_loaded:
                st.error("⚠️ Модель не загружена.")
            elif st.session_state.searcher is None:
                st.error("⚠️ База данных не загружена.")
            else:
                with st.spinner("🎧 Обработка аудиофайла..."):
                    result, error = process_audio_file(audio_path)
                    if error:
                        st.error(f"❌ Ошибка: {error}")
                    else:
                        st.session_state.result = result
                        st.success("✅ Готово! Результат внизу.")
        
        if os.path.exists(audio_path):
            os.unlink(audio_path)

# ===== РЕЗУЛЬТАТ =====
st.divider()
st.subheader("📊 Результат")

if st.session_state.result is not None:
    result = st.session_state.result
    
    if "error" in result:
        st.error(f"❌ {result['error']}")
        if "raw" in result:
            with st.expander("📝 Сырой ответ модели"):
                st.code(result['raw'])
    else:
        product = result.get('product', 'Не указано')
        hs_code = result.get('hs_code', 'Не найден')
        found = result.get('found_in_excel', False)
        
        st.success("✅ Товар распознан!")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("📦 Товар", product)
        with col_b:
            st.metric("🏷️ Код ТНВЭД", hs_code if hs_code else "❌ Не найден")
        
        if found:
            with st.expander("📄 Найденные совпадения"):
                st.json(result.get('all_matches', []))
        else:
            st.warning("⚠️ Код не найден в базе данных")
else:
    st.info("⏳ Скажите или загрузите аудио.")

with st.expander("💡 Примеры запросов"):
    st.markdown("""
    - **Гайки стальные диаметром 10 мм** → `7318163009`
    - **Двигатели турбореактивные** → `8411128000`
    - **Фитинги для труб** → `7307291009`
    """)

st.divider()
st.caption("🔬 Экспериментальный проект. Точность зависит от качества аудио и базы ТНВЭД.")