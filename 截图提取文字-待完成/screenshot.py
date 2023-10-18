# coding:utf-8

import time
import tkinter as tk
import keyboard
from PIL import ImageGrab, ImageTk, Image

import py_tool


# 截图类
class Screenshot:
    def __init__(self, top, scale):
        """---------------------------------------------------系统设置"""
        # 获取屏幕缩放比例;
        self.scale = scale
        # 排除缩放干扰;
        py_tool.eliminate_scaling_interference()
        # 初始化关闭窗口标志;
        self.win_close = 0
        "---------------------------------------------------初始窗口"
        # 最小化主窗口;
        top.iconify()
        # 实例化主窗口;
        self.top_screenshot = tk.Toplevel(top)
        # 不显示标题栏;
        self.top_screenshot.overrideredirect(True)
        # 窗口长和宽;
        self.width_win = self.top_screenshot.winfo_screenwidth()
        self.height_win = self.top_screenshot.winfo_screenheight()
        # 设置主窗口大小和位置;
        self.top_screenshot.geometry(str(int(self.width_win)) + 'x' + str(int(self.height_win)) + '+' +
                                     str(int((self.top_screenshot.winfo_screenwidth() - self.width_win) / 2)) + '+' +
                                     str(int((self.top_screenshot.winfo_screenheight() - self.height_win) / 2)))
        # 设置窗口背景颜色,鼠标样式;
        self.top_screenshot.config(bg='', cursor='crosshair')
        "---------------------------------------------------全屏截图"
        # 实时放大会把雾窗口和截图边框放大,不美观,速度慢,放大已完成截图的图片;
        time.sleep(0.5)
        # 全屏截图;
        self.img_screenshot = ImageGrab.grab(
            (0, 0, self.top_screenshot.winfo_screenwidth(), self.top_screenshot.winfo_screenheight()))
        # 保存截图;
        self.img_screenshot.save('test00.png')
        # 打开截图;
        self.img_screenshot = Image.open('test00.png')
        "---------------------------------------------------窗口边框"
        # 初始化边框画布边框宽度;
        self.canvas_frame_frame_width = 1 * self.scale
        # 创建上边框画布;
        canvas_frame_up = tk.Canvas(self.top_screenshot, width=self.width_win, height=self.canvas_frame_frame_width,
                                    bg='yellow',
                                    highlightthickness=self.canvas_frame_frame_width,
                                    highlightbackground='yellow')
        # 设置上边框画布位置;
        canvas_frame_up.place(relx=0.5, rely=0, anchor=tk.CENTER)
        canvas_frame_down = tk.Canvas(self.top_screenshot, width=self.width_win,
                                      height=self.canvas_frame_frame_width + 1,
                                      bg='yellow',
                                      highlightthickness=self.canvas_frame_frame_width,
                                      highlightbackground='yellow')
        canvas_frame_down.place(relx=0.5, rely=1, anchor=tk.CENTER)
        canvas_frame_left = tk.Canvas(self.top_screenshot, width=self.canvas_frame_frame_width, height=self.height_win,
                                      bg='yellow',
                                      highlightthickness=self.canvas_frame_frame_width,
                                      highlightbackground='yellow')
        canvas_frame_left.place(relx=0, rely=0.5, anchor=tk.CENTER)
        canvas_frame_right = tk.Canvas(self.top_screenshot, width=self.canvas_frame_frame_width + 1,
                                       height=self.height_win,
                                       bg='yellow',
                                       highlightthickness=self.canvas_frame_frame_width,
                                       highlightbackground='yellow')
        canvas_frame_right.place(relx=1, rely=0.5, anchor=tk.CENTER)
        "---------------------------------------------------黑雾窗口"
        # 创建独立主窗口的新窗口,雾窗口;
        self.fog = tk.Toplevel(self.top_screenshot)
        self.fog.overrideredirect(True)
        self.fog.config(bg='black')
        # 设置透明颜色;
        self.fog.wm_attributes('-transparentcolor', 'red')
        # 设置窗口透明度;
        self.fog.wm_attributes('-alpha', 0)
        self.fog.geometry(str(int(self.width_win)) + 'x' + str(int(self.height_win)) + '+' +
                          str(int((self.top_screenshot.winfo_screenwidth() - self.width_win) / 2)) + '+' +
                          str(int((self.top_screenshot.winfo_screenheight() - self.height_win) / 2)))
        "---------------------------------------------------截图画布"
        # 创建截图画布;
        self.canvas_screenshot = tk.Canvas(self.fog, bg='black', highlightthickness=0)
        # 使画布自动充满窗口;
        self.canvas_screenshot.pack(fill=tk.BOTH, expand=tk.Y)
        "---------------------------------------------------截图区域"
        # 初始化截图
        self.screenshot_img = None
        # 初始化截图区域初始 X,Y 坐标;
        self.first_x = 0
        self.first_y = 0
        # 创建矩形,充当截图区域;
        self.canvas_screenshot.create_rectangle(0, 0, 0, 0, fill='red', outline='yellow',
                                                width=round(self.canvas_frame_frame_width), tags='L')
        "---------------------------------------------------放大窗口"
        self.enlarge = tk.Toplevel(self.fog)
        self.enlarge.overrideredirect(True)
        # 设置窗口置顶;
        self.enlarge.wm_attributes('-topmost', 1)
        "---------------------------------------------------放大画布"
        # 初始画放大图片;
        self.img_tkinter = None
        # 初始化放大画布长宽;
        self.canvas_enlarge_width = round(150 * self.scale)
        self.canvas_enlarge_height = round(150 * self.scale)
        # 创建放大画布;
        self.canvas_enlarge = tk.Canvas(self.enlarge, width=self.canvas_enlarge_width,
                                        height=self.canvas_enlarge_height,
                                        highlightthickness=1, highlightbackground="yellow")
        self.canvas_enlarge.pack(fill=tk.BOTH, expand=tk.YES)
        # 间隔固定时间执行指定方法;
        self.enlarge.after(func=self.img_enlarge(), ms=10)
        "---------------------------------------------------键盘监听"
        # 监听所有键盘,并执行参数中方法;
        keyboard.hook(self.win_xit_and_keyboard_control_mouse)
        "---------------------------------------------------鼠标监听"
        # 绑定鼠标左键按下事件;
        self.top_screenshot.bind("<ButtonPress-1>", self.mouse_press_event)
        # 绑定鼠标左键拖动事件;
        self.top_screenshot.bind("<B1-Motion>", self.mouse_drag_event)
        # 绑定鼠标左键松开事件;
        self.top_screenshot.bind("<ButtonRelease-1>", self.mouse_lift_event)
        "---------------------------------------------------窗口监听"
        # 监控每个组件,当组件发生变化或触发事件时,会立即更新窗口;
        self.top_screenshot.mainloop()

    # 鼠标左键按下事件;
    def mouse_press_event(self, event):
        self.fog.wm_attributes('-alpha', 0.5)
        self.top_screenshot.wm_attributes('-alpha', 0)
        mouse_x, mouse_y = py_tool.get_mouse_coordinate()
        self.first_x = mouse_x
        self.first_y = mouse_y
        # 获取截图区域初始位置;
        self.canvas_screenshot.create_rectangle(0, 0, 0, 0, fill='red', outline='yellow',
                                                width=round(self.canvas_frame_frame_width), tags='L')
        self.canvas_screenshot.coords('L', self.first_x, self.first_y, mouse_x, mouse_y)
        return event

    # 鼠标左键拖动事件;
    def mouse_drag_event(self, event):
        mouse_x, mouse_y = py_tool.get_mouse_coordinate()
        self.canvas_screenshot.coords('L', self.first_x, self.first_y, mouse_x, mouse_y)
        return event

    # 鼠标左键抬起事件;
    def mouse_lift_event(self, event):
        self.fog.wm_attributes('-alpha', 0)
        self.top_screenshot.wm_attributes('-alpha', 1)
        mouse_x, mouse_y = py_tool.get_mouse_coordinate()
        down_right_x = 0
        down_right_y = 0
        # 获取截图左上角坐标;
        screenshot_top_left_x = self.fog.winfo_rootx() + self.first_x
        screenshot_top_left_y = self.fog.winfo_rooty() + self.first_y
        # 鼠标起始向左上截图;
        if self.first_x - mouse_x > 0 and self.first_y - mouse_y > 0:
            # 获取截图右下角坐标;
            down_right_x = screenshot_top_left_x - abs(self.first_x - mouse_x)
            down_right_y = screenshot_top_left_y - abs(self.first_y - mouse_y)
            self.screenshot_img = self.img_screenshot.crop((down_right_x, down_right_y, screenshot_top_left_x,
                                                            screenshot_top_left_y))
        # 鼠标起始向左下截图;
        if self.first_x - mouse_x > 0 > self.first_y - mouse_y:
            down_right_x = screenshot_top_left_x - abs(self.first_x - mouse_x)
            down_right_y = screenshot_top_left_y + abs(self.first_y - mouse_y)
            self.screenshot_img = self.img_screenshot.crop((down_right_x, screenshot_top_left_y, screenshot_top_left_x,
                                                            down_right_y))
        # 鼠标起始向右上截图;
        if self.first_x - mouse_x < 0 < self.first_y - mouse_y:
            down_right_x = screenshot_top_left_x + abs(self.first_x - mouse_x)
            down_right_y = screenshot_top_left_y - abs(self.first_y - mouse_y)
            self.screenshot_img = self.img_screenshot.crop((screenshot_top_left_x, down_right_y, down_right_x,
                                                            screenshot_top_left_y))
        # 鼠标起始向右下截图;
        if self.first_x - mouse_x < 0 and self.first_y - mouse_y < 0:
            down_right_x = screenshot_top_left_x + abs(self.first_x - mouse_x)
            down_right_y = screenshot_top_left_y + abs(self.first_y - mouse_y)
            self.screenshot_img = self.img_screenshot.crop((screenshot_top_left_x, screenshot_top_left_y, down_right_x,
                                                            down_right_y))
        # 保存截图;
        if down_right_x > 0 and down_right_y > 0:
            self.screenshot_img.save('test00.png')
        self.canvas_screenshot.delete('L')
        return event

    # 获取鼠标附近位置屏幕快照并放大;
    def img_enlarge(self):
        if self.win_close == 0:
            # 获取鼠标坐标;
            mouse_x, mouse_y = py_tool.get_mouse_coordinate()
            # 鼠标附近范围;
            mouse_range = 20 * self.scale
            # 抓取鼠标附近屏幕快照;
            img_cropping = self.img_screenshot.crop((mouse_x - mouse_range, mouse_y - mouse_range,
                                                     mouse_x + mouse_range, mouse_y + mouse_range))
            # 放大鼠标附近屏幕快照;
            img_mouse_nearby_enlarge = img_cropping.resize((int(self.canvas_enlarge_width),
                                                            int(self.canvas_enlarge_height)))
            # 转换为 tkinter 可用的图片;
            self.img_tkinter = ImageTk.PhotoImage(img_mouse_nearby_enlarge)
            # 画布中显示放大后的图片,参数1和2是图片左上角位置;
            self.canvas_enlarge.create_image(int(self.canvas_enlarge_width) / 2, int(self.canvas_enlarge_height) / 2,
                                             image=self.img_tkinter)
            # 画布中画线;
            self.canvas_enlarge.create_line(int(self.canvas_enlarge_width) / 2, 0, int(self.canvas_enlarge_width) / 2,
                                            self.canvas_enlarge_height, fill="yellow")
            self.canvas_enlarge.create_line(0, int(self.canvas_enlarge_width) / 2, self.canvas_enlarge_width,
                                            int(self.canvas_enlarge_height) / 2, fill="yellow")
            # 更新窗口位置;
            if mouse_x + 20 + int(self.canvas_enlarge_width) < self.top_screenshot.winfo_screenwidth() \
                    and mouse_y + 20 + int(self.canvas_enlarge_height) < self.top_screenshot.winfo_screenheight():
                self.enlarge.geometry(str(self.canvas_enlarge_width) + 'x' + str(self.canvas_enlarge_height) + '+{}+{}'
                                      .format(mouse_x + 20, mouse_y + 20))
            if mouse_x + 20 + int(self.canvas_enlarge_width) > self.top_screenshot.winfo_screenwidth() \
                    and mouse_y + 20 + int(self.canvas_enlarge_height) < self.top_screenshot.winfo_screenheight():
                self.enlarge.geometry(str(self.canvas_enlarge_width) + 'x' + str(self.canvas_enlarge_height) + '+{}+{}'
                                      .format(mouse_x - 20 - int(self.canvas_enlarge_width), mouse_y + 20))
            if mouse_x + 20 + int(self.canvas_enlarge_width) < self.top_screenshot.winfo_screenwidth() \
                    and mouse_y + 20 + int(self.canvas_enlarge_height) > self.top_screenshot.winfo_screenheight():
                self.enlarge.geometry(str(self.canvas_enlarge_width) + 'x' + str(self.canvas_enlarge_height) + '+{}+{}'
                                      .format(mouse_x + 20, mouse_y - 20 - int(self.canvas_enlarge_height)))
            if mouse_x + 20 + int(str(self.canvas_enlarge_width)) > self.top_screenshot.winfo_screenwidth() \
                    and mouse_y + 20 + int(str(self.canvas_enlarge_height)) > self.top_screenshot.winfo_screenheight():
                self.enlarge.geometry(str(self.canvas_enlarge_width) + 'x' + str(self.canvas_enlarge_height) + '+{}+{}'
                                      .format(mouse_x - 20 - int(self.canvas_enlarge_width),
                                              mouse_y - 20 - int(self.canvas_enlarge_height)))
            self.enlarge.after(func=self.img_enlarge, ms=10)
        else:
            # 退出程序;
            self.top_screenshot.destroy()

    def win_xit_and_keyboard_control_mouse(self, event):
        py_tool.keyboard_control_mouse(event)
        if keyboard.is_pressed('esc'):
            self.win_close = 1
        return event

