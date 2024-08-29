""""
1.安装SpeechRecognition
pip install SpeechRecognition

2.安装PocketSphinx
recognize_sphinx()语音识别器可以脱机工作,但是必须安装pocketsphinx库.
pip install pocketsphinx

3.下载中文模型
pocketsphinx需要安装中文语言、声学模型。
下载地址：
https://sourceforge.net/projects/cmusphinx/files/Acoustic%20and%20Language%20Models/Mandarin/

4.测试前准备
把cmusphinx-zh-cn-5.2.tar.gz下载解压后
要将文件夹更名为zh-CN,里面的zh_cn.cd_cont_5000文件夹重命名为acoustic-model.

zh_cn.lm.bin命名为language-model.lm.bin.zh_cn.dic要改为pronounciation-dictionary.dict(注意,dic改成了dict)

完成这些改名后,把这个文件夹移动到python目录下的\site-packages/speech_recognition/pocketsphinx-data/。

5.树莓派连接麦克风,运行该.py文件测试
"""



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