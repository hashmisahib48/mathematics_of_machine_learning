"""Sidebar boxes for the callout notes of every listing, read from build/ch03.js."""
import sys, os, re
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from callbox import render

def conv(t):
    t = t.replace('*X̃*^{T}*X̃*', r'$\tilde X^{\mathsf{T}}\tilde X$').replace('Σ x̄_{*j*}*w*_{*j*}', r'$\Sigma\,\bar x_j w_j$')
    t = re.sub(r'\*\*(\w)\*\*', r'$\\mathbf{\1}$', t)
    t = re.sub(r'\*(\w)\*\^\{(\w)\}', r'$\1^\2$', t)
    t = re.sub(r'\*(\w)\*', r'$\1$', t)
    return t

src = open(os.path.join(HERE, '..', 'build', 'chapter.js')).read()
for m in re.finditer(r"add\((?:listing|codeOnly)\('([\d.]+)', '([^']+)'.*?\[\n(.*?)\]\)\)", src, re.S):
    no, file, body = m.groups()
    notes = [(k + 1, '', conv(t)) for k, t in enumerate(re.findall(r"^\s*'(.*)',?$", body, re.M))]
    h = render('CD', notes, 'Listing ' + no, file, os.path.join(HERE, 'png', f'call_{no}.png'))
    print(no, len(notes), round(h, 2))
