import matplotlib.font_manager as fm, matplotlib.pyplot as plt, os
for f in ('NotoSansCJK-Regular.ttc', 'NotoSansCJK-Medium.ttc', 'NotoSansCJK-Light.ttc'):
    p = '/usr/share/fonts/opentype/noto/' + f
    if os.path.exists(p):
        fm.fontManager.addfont(p); name = fm.FontProperties(fname=p).get_name(); break
plt.rcParams['font.family'] = [name, 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'dejavusans'
