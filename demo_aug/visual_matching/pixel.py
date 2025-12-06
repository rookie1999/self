import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
matplotlib.use('TkAgg')
# 读取图片 (请替换为您的图片路径)
image_path = 'pic.jpg'
img = mpimg.imread(image_path)

plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.title("Click on the image to get coordinates")

# 方式 A: 鼠标悬停查看
# 运行后，鼠标放在图片上，窗口右下角会自动显示 x=..., y=...

# 方式 B: 点击获取坐标 (获取 5 个点)
print("请在图片上点击 5 个点...")
points = plt.ginput(5)
print("您点击的坐标为:", points)

plt.show()