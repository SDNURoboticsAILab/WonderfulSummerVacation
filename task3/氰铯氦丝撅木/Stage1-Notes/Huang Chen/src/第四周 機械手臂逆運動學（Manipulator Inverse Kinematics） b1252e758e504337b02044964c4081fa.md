# 第四周 機械手臂逆運動學（Manipulator Inverse Kinematics）

# 引言

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image.png)

# 求解概念

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%201.png)

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%202.png)

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%203.png)

# 多重解

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%204.png)

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%205.png)

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%206.png)

# 求解方法

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%207.png)

# Example: A RRR Manipulator

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%208.png)

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%209.png)

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%2010.png)

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%2011.png)

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%2012.png)

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%2013.png)

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%2014.png)

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%2015.png)

# 三角函數方程式求解

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%2016.png)

# Pieper’s Solution

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%2017.png)

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%2018.png)

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%2019.png)

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%2020.png)

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%2021.png)

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%2022.png)

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%2023.png)

# 座標系

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%2024.png)

# Example: 物件取放任務

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%2025.png)

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%2026.png)

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%2027.png)

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%2028.png)

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%2029.png)

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%2030.png)

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%2031.png)

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%2032.png)

![image.png](%E7%AC%AC%E5%9B%9B%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%80%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Inverse%20Kinematics%EF%BC%89%20b1252e758e504337b02044964c4081fa/image%2033.png)

# 「4-4 物件取放任務」 教材勘誤補充

目前投影片錯誤皆已修正，但因影片處理較費時，麻煩大家先對照勘誤表。如果還有其他錯誤敬請指教。

「4-4 物件取放任務」：

1. 影片13:43-15:03桌子座標標示原點錯誤，長度250應由桌子原點起算。可以參考7:55-13:42的標示方法。（投影片在p28, 30, 31）

2. 影片16:18-17:21 r=1.68498214e5修正為168813.18。（投影片 p33）

3. 後三軸Euler angle的計算還有其他細節須注意，請見下方補充。

關於本週教材及作業內容，Example及Quiz中的手臂構型由於後三軸相交於一點，得以獨立拆解成兩部分計算，可注意以下幾點：

[前三軸]

手臂IK本身具有多重解且計算細節多，建議搭配FK確認是計算錯誤抑或只是求到另一組解。以教材中的Example來說，前三軸共有四組解：(21.8, -52.2, 2.5), (21.8, 47.1, 164.0), (201.7, 126.9, 15.2), (201.7, -121.1, 151.3)，關於這四組解的構型，可以參考投影片p7，或者影片「4-2_多重解1」1:34-4:42。至於Quiz部分請依照題目要求的角度限制作答。關於不同解法的補充如下：

1. Pieper’s solution

Pieper’s solution方程式較複雜且非線性程度高，建議使用MATLAB或其他軟體計算。以MATLAB的solve為例，因其對非線性方程式之處理能力較有限，建議參考影片「4-2_多重解 3 Example 2」9:22-end或投影片 p18，將三角函數用subs指令替換成多項式，進行後續計算。

另外，由於列式時將部分距離及三角函數取平方，可能會算出反向距離的解，請搭配FK確認排除。

2. 幾何法

使用幾何法則須想像出手臂到達該點的姿態，才能列出所有的可行解。列式時請注意手臂的offset項。

[後三軸]

順利取得前三軸座標後，可以計算出30𝑅30​*R*及63𝑅63​*R*，如此便能使用week2所學將rotation matrix逆向拆解出𝜃4,𝜃5,𝜃6*θ*4​,*θ*5​,*θ*6​。要注意的是，必須確保手臂姿態與所使用解法重合，若使用教材中所提的ZYZ Euler angle解之，就需再將𝑍3*Z*3​轉到與𝑍4*Z*4​重合。進一步的說明可以參考這篇的staff的留言：

[https://www.coursera.org/learn/robotics1/discussions/weeks/4/threads/YCt2UUpmEeecxwq_51gzqg](https://www.coursera.org/learn/robotics1/discussions/weeks/4/threads/YCt2UUpmEeecxwq_51gzqg)

或者是第七周相關內容的教材說明：

[https://www.coursera.org/learn/robotics1/lecture/GLmaG/7-2-gui-ji-gui-hua-shi-li-fang-fa-2](https://www.coursera.org/learn/robotics1/lecture/GLmaG/7-2-gui-ji-gui-hua-shi-li-fang-fa-2)

[https://www.coursera.org/learn/robotics1/lecture/GLmaG/7-2-gui-ji-gui-hua-shi-li-fang-fa-2](https://www.coursera.org/learn/robotics1/lecture/GLmaG/7-2-gui-ji-gui-hua-shi-li-fang-fa-2)