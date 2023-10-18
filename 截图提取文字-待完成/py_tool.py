from ctypes import windll

import keyboard
import win32api
import win32con
import win32gui
import win32print
from pynput import mouse


# 获取鼠标坐标;
def get_mouse_coordinate():
    control = mouse.Controller()
    mouse_x, mouse_y = control.position
    return mouse_x, mouse_y


# 获取屏幕缩放比例;
def get_screen_scale_rate():
    # 获取屏幕实际宽度
    hdc = win32gui.GetDC(0)
    width_actual_screen = win32print.GetDeviceCaps(hdc, win32con.DESKTOPHORZRES)
    # 获取屏幕缩放宽度
    width_zoom_screen = win32api.GetSystemMetrics(0)
    # 计算缩放比率
    screen_scale_rate = round(width_actual_screen / width_zoom_screen, 2)
    return screen_scale_rate


# 排除缩放干扰;
def eliminate_scaling_interference():
    windll.user32.SetProcessDPIAware()


# 实现键盘控制鼠标;
def keyboard_control_mouse(event):
    # 创建鼠标实例,用于键盘移动鼠标;
    control = mouse.Controller()
    if keyboard.is_pressed('up'):
        control.move(0, -1)
    if keyboard.is_pressed('down'):
        control.move(0, 1)
    if keyboard.is_pressed('left'):
        control.move(-1, 0)
    if keyboard.is_pressed('right'):
        control.move(1, 0)
    if keyboard.is_pressed('space'):
        control.press(mouse.Button.left)
    if keyboard.is_pressed('enter'):
        control.release(mouse.Button.left)
    return event

