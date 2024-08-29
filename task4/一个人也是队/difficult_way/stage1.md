stage1.连接并且使用树莓派
这个文档集合了我连接树莓派遇到的难点，可以简单看一看

A：系统的烧录
我们要是想去利用树莓派，首要就是连接树莓派。
一般拿到树莓派最好先烧录一下，重置一下树莓派
树莓派烧录很简单，读卡器读取树莓派，用官方app烧录即可，贴一个官方的教程
https://blog.csdn.net/lx_nhs/article/details/124859914


2：树莓派的连接
首先先用充电器给树莓派供电，再将网线连接电脑和主机
用网线的话,找树莓派ip和通过ssh连接有点麻烦
有俩个问题需要解决

    question1:首先将自己网络设置成可共享的，在我们用网线连接其它设备的时候，
我们都需要设置这个,反正我当时连接cyberdog2也是需要的。
讲一个windows下的设置,
window + r 输入control回车 -> 网络和internet -> 网络和共享中心 -> 更改适配器设置 ->
双击wifi -> 属性 -> 共享 -> 第二个"允许什么什么"打勾就可以了.    
wifi设置旁边的以太网, 如果有设备接入, 也可以看到网线口的ip记住一会有用.
双击详细信息的ipv4就是
    tips：(有时候查找不到树莓派ip需要重复一下这个步骤)

    question2：树莓派一开始是拒绝ssh连接, 这就意味着我们无法访问它, 因此我们需要读卡器读取sd卡
并且在其中创建一个名字为ssh无后缀名字的文件。
    tips：（而且这个文件不知道为什么有时候会自动消失，可能是树莓派自动清理所以如果监测不到ip，需要再用读卡器重新创建ssh文件。）
解决完就可以连接了   
用windows + r 输入cmd打开终端, 输入arp -a看所有ip从上边找到网线口ip的第一个(非255)
ssh连接即可，如果你的windows不支持ssh,可以下载一个putty来连接

node2:树莓派一开始的源是树莓派那边的不能用。
因此我们最好考虑一下换源，这样就可以利用 apt 和 pip 安装一些必要的东西
    贴一个简单的教程：
https://blog.csdn.net/qq_50827004/article/details/133754825
    tips：(树莓派只有nano, 很多小伙伴不会用nano修改文件。讲一个最简单的操作
首先ctrl + o保存, 然后记得回车确认!最后ctrl + x退出。)

node3:树莓派可视化
    贴个教程(语音识别其实用不到可视化QAQ)
https://blog.csdn.net/weixin_42108484/article/details/103820532?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1.pc_relevant_default&spm=1001.2101.3001.4242.2&utm_relevant_index=4