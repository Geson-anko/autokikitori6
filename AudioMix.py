from pydub import AudioSegment

def testmix():
    sound0 = AudioSegment.from_file('data/voice_only/4704.wav')
    sound1 = AudioSegment.from_file('data/voice_only/common_voice_en_1000.wav')

    output = sound0.overlay(sound1,position=0)
    output.export("testmix.wav",format="wav")

if __name__ == '__main__':
    testmix()