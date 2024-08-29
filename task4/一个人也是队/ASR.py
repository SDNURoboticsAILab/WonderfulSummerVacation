import speech_recognition as sr
 
def asr(rate=16000):
   r = sr.Recognizer()
 
   with sr.Microphone(sample_rate=rate) as source:
      print("说些什么…… ")
      audio = r.listen(source)
 
   try:
     c=r.recognize_sphinx(audio, language='zh-CN')
    #c=r.recognize_sphinx(audio, language='en-US')
     print("识别结果: " + c)
   except sr.UnknownValueError:
      print("无法识别音频")
   except sr.RequestError as e:
      print("Sphinx error; {0}".format(e))
 
asr()