%%%%%%%%%%%%%%%%%%%%%%%%%%
% This Class is for the Tesi Style of the University
% of Florence.
% The book LaTeX2e class is used and modified.
% F. Bartolini, DIE, via S. Marta 3, 50139, Firenze
% barto@cosimo.ing.unifi.it
%
%  You need LaTeX2e with the graphics bundle installed.
%  You also need the babel package installed for getting
%      italian hyphenation and names
%
%   Usage:
%       \documentclass[twoside|oneside,10pt|11pt|12pt]{TPtesi}
%   Default:
%       \documentclass[oneside,12pt]{TPtesi.cls}
%
%   Compulsory commands:
%       \title{......}             Thesis title
%       \author{......}            Thesis author
%       \titolocorso{......}       Laurea or doctorate course name
%       \chair{.......}            First thesis supervisor
%       \numberofmembers{..}       Number of supervisors (at least 1)
%       \degreeyear{......}        Accademic year of graduation
%   Available commands:
%       \othermembers{.....\mediskip\\....} Possible other supervisors
%%%%%%%%%%%%%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{Tptesi2}
     [23/2/96 v1.0 Classe Latex per tesi  ---
       basata su UFtesi.cls by F. Bartolini, Dip. Ing. Elettronica -  Univ. Firenze]
\newcommand{\@ptoption}{}
\newcommand{\@tipo}{}
\DeclareOption{10pt}{\renewcommand{\@ptoption}{0pt}}
\DeclareOption{11pt}{\renewcommand{\@ptoption}{1pt}}
\DeclareOption{12pt}{\renewcommand{\@ptoption}{2pt}}
\DeclareOption{twoside}{\renewcommand{\@tipo}{twoside}}
\DeclareOption{oneside}{\renewcommand{\@tipo}{oneside}}
\ExecuteOptions{oneside,12pt}
\ProcessOptions\relax
%
% Load the book class for a4 paper dimension
%
\LoadClass[\@tipo,a4paper,1\@ptoption]{book}
% The babel package is required for italian.
\RequirePackage[italian]{babel}
% The amsmath package is required for good-looking formulas.
\RequirePackage[reqno]{amsmath}
% The graphicx package is required for inserting the University
% logo in the title page.
\RequirePackage{amssymb}
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{float}
\RequirePackage[linesnumbered,boxed,noend]{algorithm2e}
\RequirePackage{mathrsfs}

% >>>>>>>>>>------------------------------------------<<<<<<<<<
% The following part was added by Nicolo' Giorgetti, 2002/02/13
% It is useful with pdflatex and dvipdfm
\ifx\pdfoutput\undefined
     \RequirePackage[dvips]{graphicx}
     \newcommand{\myurl}[1]{\url{#1}}
     %\RequirePackage[dvipdfm,colorlinks,hyperindex]{hyperref}
\else
     \RequirePackage[pdftex]{graphicx}
     \DeclareGraphicsExtensions{.pdf, .png, .jpg, .mps}
     \RequirePackage[colorlinks, pdfpagelabels, hyperindex]{hyperref}
     \newcommand{\myurl}[1]{\href{#1}{\url{#1}}}
     % Decomment below row if you want the thumbnails in your
     % PDF document
     %\usepackage{thumbpdf}
%\fi
% >>>>>>>>>>------------------------------------------<<<<<<<<<


% The chapterbib package is required for the bibliograpy.
% \RequirePackage{chapterbib}
% The subfigure package is required for the subfigures.
% \RequirePackage{subfigure}
%
% The following variables are defined for the thesis title page
% The name of the laurea course
\def\titolocorso#1{\gdef\@titolocorso{#1}}
% The Academic year the degree will be officially conferred
\def\degreeyear#1{\gdef\@degreeyear{#1}}
% The name of your committee's chair
\def\chair#1{\gdef\@chair{#1}}
% The names of your other committe members, one per line
\def\othermembers#1{\gdef\@othermembers{#1}}
% correlatori
\def\correlatori#1{\gdef\@correlatori{#1}}
% The number of committee members, which affects both the
% number of lines and the amount of space between lines
% on the approval page.
\def\@numberofmembers{3}
\def\@approvalspace{.75in}
\def\numberofmembers#1{\gdef\@numberofmembers{#1}
\ifnum \@numberofmembers > 3
  \gdef\@approvalspace{.5in}
\fi}
% numero di correlatori
\def\@numerocorrelatori{3}
\def\numerocorrelatori#1{\gdef\@numerocorrelatori{#1}
\ifnum \@numerocorrelatori > 3
  \gdef\@approvalspace{.5in}
\fi}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%   TPtesi   %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\linespacing}{1.5}
\renewcommand{\baselinestretch}{\linespacing}
\newcommand{\rilegatura}{10mm}
\addtolength\oddsidemargin{\rilegatura}
\addtolength\evensidemargin{-\rilegatura}
\setlength{\headsep}{15mm}
% begin single line spacing command
\def\BeginSingle{\def\baselinestretch{1.0}\large\normalsize}
% restore normal line spacing command
\def\EndSingle{\par\def\baselinestretch{\linespacing}\large\normalsize}
%%%%%%%%%%%%%%%%%%%

% The abstract which is not present in the original
% LaTeX2e book class is inserted.
% The abstract definition is taken from the report class.
%
\if@titlepage
  \newenvironment{abstract}{%
      \titlepage
      \null\vfil
      \begin{center}
        \bfseries \abstractname
      \end{center}}
     {\par\vfil\null\endtitlepage}
\else
  \newenvironment{abstract}{%
      \if@twocolumn
        \section*{\abstractname}%
      \else
        \small
        \begin{center}%
          {\bfseries \abstractname\vspace{-.5em}\vspace{\z@}}%
        \end{center}%
        \quotation
      \fi}
      {\if@twocolumn\else\endquotation\fi}
\fi

% The titlepage environment is changed so that
% a 1 \baselinestretch is used.
% The global \baselinestretch (as defined by
% the \linespacing command above) is then restored.
%
\renewenvironment{titlepage}
    {%
      \if@twocolumn
        \@restonecoltrue\onecolumn
      \else
        \@restonecolfalse\newpage
      \fi
      \renewcommand{\baselinestretch}{1.0}
      \thispagestyle{empty}%
      \if@compatibility
        \setcounter{page}{0}
      \else
        \setcounter{page}{1}%
      \fi}%
    {\if@restonecol\twocolumn \else \newpage \fi
      \renewcommand{\baselinestretch}{\linespacing}
      \setcounter{page}{1}%
    }

% The \maketitle command is adapted to build a
% title page suitable for the University of Florence.
%
\renewcommand{\maketitle}
{\begin{titlepage}%
  \let\footnotesize\small
  \let\footnoterule\relax
  \BeginSingle
  \vspace*{-15mm}
  \centerline{\includegraphics[height=30mm]{./img/stemma}}
  \begin{center}
    \textsc{\Large Universit\`a degli Studi di Firenze}\medskip\\
    \textsc{ Scuola di Ingegneria - Dipartimento di Ingegneria dell'Informazione}\medskip\\
    \rule{50mm}{0.01mm}\medskip\\
    {\small Tesi di Laurea Triennale in}
    \@titolocorso\medskip\\
    \vfill
    {\Large \bf \vspace{2cm} \textsc{\@title} \par}
    \vspace{2cm}

\par
  \end{center}\par
 \vbox to 65mm {
  \hfill
 \begin{minipage}[t]{65mm}% Autore sulla Sx
   \raggedright
   {\it Candidato}\newline
   \@author
 \end{minipage}%
 \begin{minipage}[t]{60mm}% Relatori sulla Dx
  \ifnum \@numberofmembers > 1
    {\it Relatori}\newline
  \else
    {\it Relatore}\newline
  \fi
    {\@chair}  \vspace{5mm}
    \ifnum \@numberofmembers > 1
      {\@othermembers}
    \fi
  \vspace{5mm}\par
  \ifnum \@numerocorrelatori > 1
    {\it Correlatori}\newline
  \else
    {\it Correlatore}\newline
  \fi
    {\@correlatori}
 \end{minipage}%
 \hfil\vfil}%

  \vfill
  \begin{center}
    \rule{40mm}{0.01mm}\\
    {Anno Accademico \@degreeyear}
  \end{center}
  \end{titlepage}
  \setcounter{footnote}{0}%
  \let\thanks\relax\let\maketitle\relax
  \gdef\@thanks{}\gdef\@author{}\gdef\@title{}
  \EndSingle
}

% The page heading are changed. The uppercase forcing
% is avoided and a rule is inserted below the heading.
%
\if@twoside
  \def\ps@headings{%
      \let\@oddfoot\@empty\let\@evenfoot\@empty
      \def\@evenhead{\thepage\hfil\slshape\leftmark%
      \hspace{-\textwidth}\rule[-2mm]{\textwidth}{0.01mm}}%
      \def\@oddhead{{\slshape\rightmark}\hfil\thepage%
      \hspace{-\textwidth}\rule[-2mm]{\textwidth}{0.01mm}}%
      \let\@mkboth\markboth
    \def\chaptermark##1{%
      \markboth {
        \ifnum \c@secnumdepth >\m@ne
          \if@mainmatter
            \@chapapp\ \thechapter. \ %
          \fi
        \fi
        ##1}{}}%
    \def\sectionmark##1{%
      \markright {%
        \ifnum \c@secnumdepth >\z@
          \thesection. \ %
        \fi
        ##1}}}
\else
  \def\ps@headings{%
    \let\@oddfoot\@empty
    \def\@oddhead{{\slshape\rightmark}\hfil\thepage%
    \hspace{-\textwidth}\rule[-2mm]{\textwidth}{0.01mm}}%
    \let\@mkboth\markboth
    \def\chaptermark##1{%
      \markright {%
        \ifnum \c@secnumdepth >\m@ne
          \if@mainmatter
            \@chapapp\ \thechapter. \ %
          \fi
        \fi
        ##1}}}
\fi

% The caption is changed so that a \small font is used.
%
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \BeginSingle
  \sbox\@tempboxa{{\small #1: #2}}%
  \ifdim \wd\@tempboxa >\hsize
    {\small #1: #2}\par
  \else
    \global \@minipagefalse
    \hbox to\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \EndSingle
  \vskip\belowcaptionskip}
%
% The \tableofcontents command is changed so that
% the \uppercase forcing of the header is avoided
%
\renewcommand{\tableofcontents}{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\contentsname
        \@mkboth{\contentsname}{\contentsname}}%
    \@starttoc{toc}%
    \if@restonecol\twocolumn\fi
    }
%
% The \listoffigures command is changed so that
% the \uppercase forcing of the header is avoided
%
\renewcommand{\listoffigures}{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listfigurename
      \@mkboth{\listfigurename}%
              {\listfigurename}}%
    \@starttoc{lof}%
    \if@restonecol\twocolumn\fi
    }
%
% The \listoftables command is changed so that
% the \uppercase forcing of the header is avoided
%
\renewcommand{\listoftables}{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listtablename
      \@mkboth{\listtablename}{\listtablename}}%
    \@starttoc{lot}%
    \if@restonecol\twocolumn\fi
    }
% The thebibliography environment is changed so that
% the bibliography page is inserted in the list of content.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                                                                                                          %
%       \renewenvironment{thebibliography}[1]                                                                      %
%                {\chapter*{\bibname                                                                                               %
%                       \@mkboth{\bibname}{\bibname}}%                                                                     %
%                       \addcontentsline{toc}{chapter}{\bibname}%                                                  %
%                 \list{\@biblabel{\arabic{enumiv}}}%                                                              %
%                          {\settowidth\labelwidth{\@biblabel{#1}}%                                                %
%                               \leftmargin\labelwidth                                                                             %
%                               \advance\leftmargin\labelsep                                                               %
%                               \if@openbib                                                                                                        %
%                                 \advance\leftmargin\bibindent                                                            %
%                                 \itemindent -\bibindent                                                                          %
%                                 \listparindent \itemindent                                                               %
%                                 \parsep \z@                                                                                              %
%                               \fi                                                                                                                        %
%                               \usecounter{enumiv}%                                                                               %
%                               \let\p@enumiv\@empty                                                                               %
%                               \renewcommand{\theenumiv}{\arabic{enumiv}}}%                               %
%                 \BeginSingle                                                                                                             %
%                 \if@openbib                                                                                                              %
%                       \renewcommand{\newblock}{\par}                                                                     %
%                 \else                                                                                                                            %
%                       \renewcommand{\newblock}{\hskip .11em \@plus.33em \@minus.07em}%   %
%                 \fi                                                                                                                              %
%                 \sloppy\clubpenalty4000\widowpenalty4000%                                                        %
%                 \sfcode`\.=\@m}                                                                                                          %
%                {\def\@noitemerr                                                                                                          %
%                  {\@latex@warning{Empty `thebibliography'     environment}}%                     %
%                 \EndSingle                                                                                                               %
%                 \endlist}                                                                                                                        %
%                                                                                                                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The \part command is changed so that the page number is
% not inserted in the part starting page.
%
\renewcommand{\part}{\cleardoublepage
                 \thispagestyle{empty}%
                 \if@twocolumn
                     \onecolumn
                     \@tempswatrue
                   \else
                     \@tempswafalse
                 \fi
                 \hbox{}\vfil
                 \secdef\@part\@spart}
%
% The \chapter command is changed so that the page number is
% not inserted in the chapter starting page.
%
\renewcommand{\chapter}{\if@openright\cleardoublepage\else\clearpage\fi
                    \thispagestyle{empty}%
                    \global\@topnum\z@
                    \@afterindentfalse
                    \secdef\@chapter\@schapter}
%
% set single spacing in table environment
%\renewenvironment{table}
%               {\BeginSingle\@float{table}}
%               {\EndSingle\end@float}
%\renewenvironment{table*}
%               {\BeginSingle\@dblfloat{table}}
%               {\EndSingle\end@dblfloat}


%
% Comandi per figure e tabelle fisse (cfr. Beccari, 227 e segg.):
%
% \begin{sidefigure}
% ...
% \end{sidefigure}
%
% \begin{localfigure}
% ...
% \end{localfigure}
%
%
\newdimen\sideftwd
\newbox\local@box\newbox\local@hbox
\def\sidepar{\newline\hspace*{\parindent}}
\def\sidefigure{\if@twocolumn\local@ft{figure}\else\side@ft{figure}\fi}
\def\endsidefigure{\endside@ft}
\def\sidetable{\if@twocolumn\local@ft{table}\else\side@ft{table}\fi}
\def\endsidetable{\endside@ft}
%
\def\side@ft#1{\par
\sideftwd=\textwidth \divide\sideftwd by 2
\def\@captype{#1}
\setbox\@tempboxa\vtop\bgroup\textwidth=\sideftwd
\columnwidth=\sideftwd \hsize\columnwidth
\@parboxrestore}
%
\def\endside@ft{\egroup
\@tempdima=\ht\@tempboxa
\advance\@tempdima by \dp\@tempboxa
\@tempcnta=\@tempdima
\divide\@tempcnta by \baselineskip
\advance\@tempcnta by 2
\global\hangindent-\sideftwd \global\hangafter-\@tempcnta
\noindent \dp\@tempboxa=\z@ \ht\@tempboxa=\z@
\hbox to \z@{\hskip\textwidth
\hbox to \z@{\hss\box\@tempboxa}\hss}%
\hskip\parindent
\global\@ignoretrue}
%
\def\local@ft#1{\def\@captype{#1}
\setbox\local@box\vbox\bgroup
\boxmaxdepth\z@ \hsize0.9\columnwidth}
\def\endlocal@ft{\egroup
\[\hbox{\lower1ex\box\local@box}\]
\global\@ignoretrue}
%
\def\localfigure{\local@ft{figure}}
\def\endlocalfigure{\endlocal@ft}
\def\localtable{\local@ft{table}}
\def\endlocaltable{\endlocal@ft}
