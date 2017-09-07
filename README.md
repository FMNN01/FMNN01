# FMNN01

Work from the course Numerical linear algebra (FMNN01) at Lund
University.


## Formatting LaTeX

In shared projects LaTeX source is read more than it is written.
Therefore the source should be formatted for readability.

### Text Flow

Lines should not overflow 80 characters.
Long lines are not only hard to read, but also formatted poorly in
simple editors such as in a mail clients.

To make version control diffs readable, every sentence should be begin
on a new line.
For example:
```latex
I like trains.
I do so even if they are late sometimes.
```


### Sections

Before any `\{xyz}section` command there should be exactly two empty
lines.
`\{xyz}section` should be followed by exactly one newline.
This keeps paragraphs closer to the heading to which they belong, than
to other sections.
For example:
```latext
\section{Title}

Some introductory text.


\subsection{First Section}

Lorem ipsum...


\subsection{Second Section}

... dolor sit amet...
```


### Usepackage

Every included package with `\usepackage` should be on its own line.
They may be grouped in logical blocks separated by empty lines.
Whenever possible, they should be sorted lexically within each block.
These guide lines prevent accidentially adding the same package twice.
For example:
```latex
\usepackage{amsfonts}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsthm}

\usepackage[english]{babel}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
```


### Indentation

Indentation is 2 spaces.
For LaTeX source it makes no sense to have a more indentation, since it
makes the source less readable.
Blocks within all `\begin{xyz}`, where `xyz` is arbitrary, should be
indented, with the only exception for `\begin{document}`.


### Equations

Do not ever use `$$`.
(See [1].)
For brevity, use `\[` and `\]` instead of `\begin{equation*}` and
`\end{equation*}`.
Also use use `\(` and `\)` instead of `$`. (See [2].)

When equations need to be broken, indent and align the natural
separators (such as equalities, order relations, equivalences or
implications).
For example:
```latex
\[
  \rho(A)
    = |\lambda|
    = \frac{|\lambda|\|v\|}{\|v\|}
    = \frac{\|\lambda v\|}{\|v\|}
    = \frac{\|Av\|}{\|v\|}
    \le \sup_{x \ne 0} \frac{\|Ax\|}{\|x\|}
    = \|A\|.
\]
```

[1] ftp://ftp.dante.de/tex-archive/info/l2tabu/german/l2tabu.pdf  
[2] https://tex.stackexchange.com/questions/510/are-and-preferable-to-dollar-signs-for-math-mode
