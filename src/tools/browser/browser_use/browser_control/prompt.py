DESCRIPTION = """
浏览器控制工具。
用于打开网页，并按 <world><browser> 里的可点击目标 index、可滚动区域 index、或 tabs 中的 tab index 进行滚动、点击、标签页切换/关闭、坐标校准、后退/前进等操作。
打开新网页使用 action=open,url；url 省略时打开 Google 搜索页；浏览器未开启时会先启动浏览器。
浏览器使用 Agent 自己的逻辑网络：公网经隔离网关访问；Windows 本机、局域网和本地文件不可见。启用 Linux 工作区后，工作区内 localhost 服务可直接使用其报告的 http://127.0.0.1:<port>/ 或 http://localhost:<port>/，该地址不会指向 Windows 本机；工作区未启用时 localhost 为空。
切换标签页使用 action=switch_tab,index；
关闭标签页使用 action=close_tab,index；action=close_browser 直接关闭整个浏览器。

注意：
- 如果你关闭了浏览器的最后一个标签页，则等同于关闭整个浏览器。

好习惯：
- 当已经不需要用到某个标签页时，记得 close_tab。
- 当已经不需要再使用浏览器时，记得 close_browser。
"""
