import streamlit as st
import pandas as pd
import numpy as np
from resemblyzer import preprocess_wav, VoiceEncoder #pip install resemblyzer
from pathlib import Path
from st_audiorec import st_audiorec # pip install streamlit-audiorec
import soundfile as sf
import shutil
import io
from gtts import gTTS
import subprocess
import os


nsf_hifigan = False
config_path = ''

def main() :
    st.title('ECHO')
    
    tab_titles = ['모델 학습', '음성 생성하기', '목소리 유사도 테스트']
    model_tab, voice_tab, voice_similarity_test_tab = st.tabs(tab_titles)
    
    with model_tab:
        st.header(tab_titles[0])

        # model name input
        model_name = st.text_input('모델 이름을 입력해주세요.')

        # model upload
        uploader_msg = '학습을 위한 목소리 녹음 파일을 업로드해주세요. 총 분량이 1시간 이상이어야 합니다.'
        voice_files = st.file_uploader(uploader_msg, accept_multiple_files=True)

        col1, col2 = st.columns([.21, 1])
        
        # train button
        with col1:
            if st.button('모델 학습하기'):
                st.write('모델 학습 중...')
                train(model_name, voice_files)
        # open tensorboard button
        with col2:
            if st.button('TensorBoard 열기'):
                st.success('포트 6006에서 TensorBoard가 성공적으로 열렸습니다!')
                open_tensorboard(model_name)

        
    with voice_tab:
        st.header(tab_titles[1])

        # model select
        model_list = os.listdir('diff-svc/checkpoints')
        if 'nsf_hifigan' in model_list:
            nsf_hifigan = True
        
        model_except = ['hubert', '0102_xiaoma_pe', 'nsf_hifigan', '0109_hifigan_bigpopcs_hop128']
        model_list = [model for model in model_list if model not in model_except]
        model_option = st.selectbox('모델을 선택해주세요.', model_list)

        # text input
        text = st.text_input('텍스트를 입력해주세요.')
        
        # infer button
        if st.button('음성 생성하기'):
            st.write('음성 생성 중...')
            infer(model_option, text)

            # get result
            inferred_audio = open('diff-svc/results/speech.wav', 'rb') # TODO
            st.audio(inferred_audio.read(), format='audio/wav')
            st.success('음성 생성 완료!')
            
    with voice_similarity_test_tab: # 목소리 유사도 테스트
        st.header(tab_titles[2])
        
        st.subheader("목소리 녹음하기")
        
        wav_audio_data = st_audiorec()
        
        if wav_audio_data is not None:
            f = open("file1.wav", "bx")
            f.write(wav_audio_data)
            f.close()
            
        voice_sim_test_file = st.file_uploader("목소리 유사도 테스트를 진행할 파일을 업로드해 주세요.")
        
        if wav_audio_data is not None:
            data, samplerate = sf.read(io.BytesIO(wav_audio_data))
        
        if voice_sim_test_file is not None:
            voice_sim_test_file = voice_sim_test_file.getvalue()
            f = open("file2.wav", "bx")
            f.write(voice_sim_test_file)
            f.close()
            
        
        if st.button("목소리 유사도 측정하기"):
            
            # similarity
            similarity = 100 * compare_voices('file1.wav', 'file2.wav') # 이거 경로만 수정해줘
        
            # estimate
            if similarity > 85:
                estimate = 'same person'
            elif similarity > 75:
                estimate = 'quite similar'
            else:
                estimate = 'not same person'
        
            st.write(f"Similarity: {similarity:.2f}% ({estimate})")

def infer(model_name, text):
    tts = gTTS(text, lang='ko')
    tts.save('diff-svc/raw/text.wav')

    try:
        os.remove('diff-svc/infer_.py')
    except FileNotFoundError:
        pass

    with open('diff-svc/infer.py', 'rt') as fin:
        with open('diff-svc/infer_.py', 'wt') as fout:
            for line in fin:
                fout.write(line.replace('test', model_name))

    os.chdir('diff-svc')
    os.system('python infer_.py')
    os.chdir('..')


def train(model_name, voice_files):
    presets = f"""raw_data_dir: diff-svc/preprocess_out/final
        binary_data_dir: diff-svc/data/binary/{model_name}
        speaker_id: {model_name}
        work_dir: diff-svc/checkpoints/{model_name}
        max_sentences: 10
        use_amp: true"""
    
    config_type = 'config_nsf.yaml' if nsf_hifigan else 'config.yaml'
    
    src = 'workspace/'+config_type
    config_path = 'diff-svc/training/'+config_type
    shutil.copy(src, config_path)

    with open(config_path,'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(presets.rstrip('\r\n') + '\n' + content)

    os.chdir('diff-svc')
    os.system('python sep_wav.py')
    os.system(f'python preprocessing/binarize.py --config {config_path}')
    os.system(f'CUDA_VISIBLE_DEVICES=0 python run.py --config {config_path} --exp_name {model_name} --reset')


def open_tensorboard(model_name):
    subprocess.run(['tensorboard', '--load_fast=true', '--reload_interval=1', '--reload_multifile=true',
        f'--logdir=diff-svc/checkpoints/{model_name}/lightning_logs', '--port=6006'])

def compare_voices(file_path1, file_path2):
    # convert file paths to path objects
    wav_path1 = Path(file_path1)
    wav_path2 = Path(file_path2)

    # preprocess the audio files
    wav1 = preprocess_wav(wav_path1)
    wav2 = preprocess_wav(wav_path2)

    # create a voice encoder
    encoder = VoiceEncoder()

    # encode the audio files and create embeddings
    embedding1 = encoder.embed_utterance(wav1)
    embedding2 = encoder.embed_utterance(wav2)

    # calculate similarity between the two embeddings using cosine similarity
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    return similarity

if __name__ == '__main__' :
    main()