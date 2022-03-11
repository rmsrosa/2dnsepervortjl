# Configuration

## config vars
+++
prepath = "2dnsepervortjl"
content_tag = ""
ignore = ["_src/"]
+++

## book variables
+++
book_title = "Two-dimensional periodic flows"
book_subtitle = "Pseudo-spectral simulation in Julia"
book_author = "<a href=\"https://rmsrosa.github.io\">Ricardo M. S. Rosa</a>"
show_license = true
book_license = "MIT License"
license_link = "LICENSE"
book_licensees = "<a href=\"https://rmsrosa.github.io\">(Ricardo M. S. Rosa)</a>"
+++

## menu variables
+++
show_aside = true
show_github = true
github_repo = "https://github.com/rmsrosa/2dnsepervortjl"
+++

## navigation links
+++
nav_on_top = true
nav_on_bottom = true
+++

## toc variables
+++
page_numbering = true
menu = [
    "*pages/intro"
    "Mathematical Formulation" => [
        "pages/pressure-velocity"
        "pages/vorticity"
        "pages/basdevant"
        "pages/stream"
    ]
    "FFT Tools" => [
        "_src/literate/tests_basic.jl"
    ]
]
+++

## page variables
+++
show_link_bagdes = true
link_view_source = true
link_download_notebook = true
link_nbview_notebook = true
link_binder_notebook = true
website = "rmsrosa.github.io/2dnsepervortjl"
+++

## binder variables
+++
nbgitpuller_repo = "rmsrosa/2dnsepervortjl"
nbgitpuller_branch = "binderenv"
binder_application = "lab" 
+++
