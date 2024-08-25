# 第三周 機械手臂順運動學（Manipulator Forward Kinematics）

# 引言

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image.png)

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%201.png)

# 手臂幾何描述方式

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%202.png)

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%203.png)

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%204.png)

# 桿件上建立Frames

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%205.png)

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%206.png)

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%207.png)

# Denavit-Hartenberg表達法 (Craig version)

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%208.png)

# Link Transformations

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%209.png)

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%2010.png)

# Example: A RRR Manipulator

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%2011.png)

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%2012.png)

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%2013.png)

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%2014.png)

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%2015.png)

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%2016.png)

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%2017.png)

# Actuator, Joint, and Cartesian Spaces

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%2018.png)

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%2019.png)

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%2020.png)

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%2021.png)

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%2022.png)

# Denavit-Hartenberg表達法小結 (Craig version)

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%2023.png)

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%2024.png)

# Denavit-Hartenberg表達法 (Standard)

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%2025.png)

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%2026.png)

# Revisit Example: A RRR Manipulator

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%2027.png)

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%2028.png)

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%2029.png)

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%2030.png)

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%2031.png)

# Example: PUMA 560

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%2032.png)

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%2033.png)

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%2034.png)

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%2035.png)

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%2036.png)

![image.png](%E7%AC%AC%E4%B8%89%E5%91%A8%20%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82%E9%A0%86%E9%81%8B%E5%8B%95%E5%AD%B8%EF%BC%88Manipulator%20Forward%20Kinematics%EF%BC%89%204d1af5917461498ab697bb65b16de8db/image%2037.png)