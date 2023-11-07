import streamlit as st
import pandas as pd
import numpy as np
from resemblyzer import preprocess_wav, VoiceEncoder #pip install resemblyzer
from pathlib import Path
from st_audiorec import st_audiorec # pip install streamlit-audiorec
import soundfile as sf
import io
from gtts import gTTS
import os


nsf_hifigan = False

def compare_voices(file_path1, file_path2): # 살려주세요 기존 코드에 있던 Path랑 저 데이터들이랑 연결할 방법을 못찾겠어요
    
    bytes1 = file_path1
    bytes2 = io.BytesIO(file_path2.read()).read() # bytes 데이터로 통일시켜놓긴 했는데 저 밑에 # preprocess the audio files랑 이을방법을 못찾겠습니다
    
    # preprocess the audio files
    #wav1 = preprocess_wav(wav_path1) ????
    #wav2 = preprocess_wav(wav_path2) ????
    
    # create a voice encoder
    encoder = VoiceEncoder()

    # encode the audio files and create embeddings
    embedding1 = encoder.embed_utterance(bytes1)
    embedding2 = encoder.embed_utterance(bytes2)

    # calculate similarity between the two embeddings using cosine similarity
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    return similarity

def main() :
    st.title('ECHO')
    
    tab_titles = ['모델 학습', '음성 생성하기', '목소리 유사도 테스트']
    model_tab, voice_tab, voice_similarity_test_tab = st.tabs(tab_titles)
    
    with model_tab: # 모델 학습 탭
        st.header(tab_titles[0])
        
        # model name input
        model_name = st.text_input('모델 이름을 입력해주세요.')

        # model upload
        uploader_msg = '학습을 위한 목소리 녹음 파일을 업로드해주세요. 총 분량이 1시간 이상이어야 합니다.'
        voice_files = st.file_uploader(uploader_msg, accept_multiple_files=True)

        # train button
        if st.button('모델 학습하기'):
            st.write('모델 학습 중..')
            train(model_name, voice_files)

        
    with voice_tab: # 음성 생성 탭
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
            st.write('음성 생성 중..')
            infer(text)

            # get result
            inferred_audio = open('diff-svc/results/test_output.wav', 'rb') # TODO
            st.audio(inferred_audio.read(), format='audio/wav')
            st.success('음성 생성 완료!')
            
    with voice_similarity_test_tab: # 목소리 유사도 테스트
        st.header(tab_titles[2])
        
        st.subheader("목소리 녹음하기")
        
        wav_audio_data = st_audiorec()
            
        voice_sim_test_file = st.file_uploader("목소리 유사도 테스트를 진행할 파일을 업로드해 주세요.")
        
        if st.button("목소리 유사도 측정하기"):
            
            # similarity
            similarity = 100 * compare_voices(wav_audio_data, voice_sim_test_file)
        
            # estimate
            if similarity > 85:
                estimate = 'same person'
            elif similarity > 75:
                estimate = 'quite similar'
            else:
                estimate = 'not same person'
        
            st.write(f"Similarity: {similarity:.2f}% ({estimate})")

def infer(text):
    tts = gTTS(text, lang='ko')
    tts.save('workspace/tts.wav')
    # TODO


def train(model_name, voice_files):
    pass
    # TODO
        
if __name__ == '__main__' :
    main()