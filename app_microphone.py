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

with st.sidebar:
    st.header("⚙️ Настройки")
    st.markdown(f"""
    **Модель:** Gemma 3n (E2B)  
    **Режим:** Мультимодальный (аудио → JSON)  
    **Язык:** Русский  
    **Устройство:** CPU (без GPU)
    """)
    
    if st.button("🔄 Загрузить модель", type="primary"):
        st.session_state.model_loaded = False
        st.rerun()
    
    st.divider()
    st.caption("Сделано в рамках проекта по классификации ТНВЭД")

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

if not st.session_state.model_loaded:
    model, processor = load_model()
    if model is not None:
        st.session_state.model = model
        st.session_state.processor = processor
        st.session_state.model_loaded = True

# ===== ФУНКЦИЯ ОБРАБОТКИ =====
def process_audio_file(audio_path):
    print(f"🔍 process_audio_file вызвана с {audio_path}")
    print(f"📁 Файл существует: {os.path.exists(audio_path)}")
    if os.path.exists(audio_path):
        print(f"📊 Размер файла: {os.path.getsize(audio_path)} байт")
    
    try:
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        print(f"✅ Аудио загружено: {len(audio)} сэмплов, {sr} Гц")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path, "sampling_rate": sr},
                    {
                        "type": "text",
                        "text": "Переведи это аудио в текст, найди товар в тексте и определи его код ТНВЭД. Отвечай ТОЛЬКО в формате JSON: {'product': 'название товара', 'hs_code': 'код ТНВЭД'}. Если код неизвестен, верни 'hs_code': null. БЕЗ пояснений."
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
        
        print(f"📝 Ответ модели: {response_text}")
        
        try:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = response_text[start:end]
                
                # Парсим JSON с поддержкой одинарных кавычек
                try:
                    result = ast.literal_eval(json_str)
                except:
                    result = json.loads(json_str)
                
                print(f"✅ Распарсенный результат: {result}")
                return result, None
            else:
                return {"error": "JSON не найден", "raw": response_text}, None
        except Exception as e:
            return {"error": str(e), "raw": response_text}, None
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
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
        
        # === ДЕКОДИРОВАНИЕ ЧЕРЕЗ av ===
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
                    if array.shape[0] == 2 or array.shape[0] == 1:
                        array = np.mean(array, axis=0)
                    else:
                        array = np.mean(array, axis=1)
                else:
                    array = array.flatten()
                    
                audio_frames.append(array)
            
            if audio_frames:
                full_audio_data = np.concatenate(audio_frames)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    sf.write(tmp_file.name, full_audio_data, samplerate=native_sr, format='WAV', subtype='PCM_16')
                    st.session_state.audio_path = tmp_file.name
                
                st.success(f"✅ Аудио успешно декодировано! Размер: {len(audio_bytes) / 1024:.1f} KB")
            else:
                st.error("❌ Запись пустая или не содержит аудио-фреймов.")
                
        except ImportError:
            st.error("❌ Библиотека av не установлена. Установите: pip install av")
            st.session_state.audio_path = None
        except Exception as e:
            st.error(f"❌ Ошибка декодирования: {e}")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_bytes)
                st.session_state.audio_path = tmp_file.name
            st.warning("⚠️ Не удалось перекодировать. Файл сохранен «как есть».")
        
        st.audio(audio_bytes, format='audio/wav')
        st.session_state.result = None

    if st.session_state.audio_path is not None and os.path.exists(st.session_state.audio_path):
        st.write("---")
        st.write("**📁 Аудио сохранено:**")
        st.write(f"Путь: {st.session_state.audio_path}")
        
        if st.button("🔍 Определить код ТНВЭД (микрофон)", type="primary", key="mic_process"):
            if not st.session_state.model_loaded:
                st.error("⚠️ Модель не загружена.")
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
        
        if st.button("🔍 Определить код ТНВЭД (файл)", type="primary", key="file_process"):
            if not st.session_state.model_loaded:
                st.error("⚠️ Модель не загружена.")
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
        
        st.success("✅ Товар распознан!")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("📦 Товар", product)
        with col_b:
            st.metric("🏷️ Код ТНВЭД", hs_code)
        
        with st.expander("📄 Полный JSON ответ"):
            st.json(result)
else:
    st.info("⏳ Скажите или загрузите аудио.")

with st.expander("💡 Примеры запросов"):
    st.markdown("""
    - **Гайки стальные диаметром 10 мм** → `7318163009`
    - **Детское питание в баночках** → `200799`
    - **Смартфон Samsung** → `851712`
    """)

st.divider()
st.caption("🔬 Экспериментальный проект. Точность зависит от качества аудио и базы ТНВЭД.")