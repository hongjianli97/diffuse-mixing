-- Typesetting only: equations/data and all numerical code remain unchanged.
function Link(el)
  local label = pandoc.utils.stringify(el.content)
  if FORMAT:match('latex') and label:match('^[%w_./%-]+$') then
    el.content = {pandoc.RawInline('latex', '\\nolinkurl{' .. label .. '}')}
    return el
  end
end

function Code(el)
  if FORMAT:match('latex') and not el.text:find('[{}]') then
    return pandoc.RawInline('latex', '\\nolinkurl{' .. el.text .. '}')
  end
end

function CodeBlock(el)
  if el.classes:includes('mermaid') then
    return pandoc.RawBlock('latex', [[
\begin{center}
\begin{tikzpicture}[box/.style={draw,rounded corners,align=center,text width=36mm,minimum height=12mm,font=\small},>=Stealth]
\node[box] (a) {轨迹身份\\固定物质质量};
\node[box,right=8mm of a] (b) {当前邻接\\物理交换导通率};
\node[box,right=8mm of b] (c) {按时间顺序\\传播浓度};
\node[box,below=12mm of a] (f) {参考物质图\\固定距离};
\node[box,below=12mm of b] (d) {实际初场的\\掺混与耗散};
\node[box,below=12mm of c] (e) {动态 ORC\\流体解释未启用};
\draw[->] (a)--(b);\draw[->] (b)--(c);
\draw[->] (a)--(f);\draw[->] (c)--(d);\draw[->] (c)--(e);
\draw[->] (f.south)--++(0,-5mm)-|(e.south);
\node[above=4mm of b,font=\small] (g) {密度来源与扩散本构};
\draw[->] (g)--(b);
\end{tikzpicture}
\end{center}
]])
  end
end
