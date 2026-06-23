DESCRIPTION = """
浏览器控制工具。
用于打开网页，并按 <world><browser> 里的可点击目标 index、可滚动区域 index、或 tabs 中的 tab index 进行滚动、点击、标签页切换/关闭、坐标校准、后退/前进等操作。
打开新网页使用 action=open,url；url 省略时打开 Google 搜索页；浏览器未开启时会先启动浏览器。
切换标签页使用 action=switch_tab,index；
关闭标签页使用 action=close_tab,index；action=close_browser 直接关闭整个浏览器。

注意：
- 这是便捷的轻量工具，如果需要按 DOM/CSS/ARIA locator 精确查找元素、填表输入文本、按键，读取元素文本或属性、统计 locator 匹配数量等进一步操作，则需要 browser_locator 工具。
- 如果你关闭了浏览器的最后一个标签页，则等同于关闭整个浏览器。

好习惯：
- 当已经不需要用到某个标签页时，记得 close_tab。
- 当已经不需要再使用浏览器时，记得 close_browser。
"""
