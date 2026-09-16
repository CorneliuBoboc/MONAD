# PDF Utilities

## DOM Structure

language: english
encoding: utf-8
css-styles: embedded
js-script: embedded

body:
  header: h2: PDF Utilities
  main: main-container: tabs
    tab: load-document(s)
    tab: smart-split-by-chapters
    tab: split-by-page(range)s

## Description

pdfutils is a self-contained Flask app that provides utilities for processing PDF documents.
It can split PDF files by chapters (using Gemini API) or by pages/pageranges.
Aditional utilities may be added in the future.

## Remarks

When splitting by pages/pageranges, do not use AI tools at all.
When splitting by chapters, follow these rules:
- use Gemini API (model gemini-3.5-flash-lite);
- user pastes their API Key (input=password) that is subsequently stored in localStorage; they may remove previousely stored key;
- detect not just chapters, but also chapter-like front/back matter sections;
- each "chapter" should have meaningful filenames; proper chapters should be numbered according to the numbers shown in the document;

## Prompt

You are an AI coder. Please write pdfutils app according to this spec.md.
If you are not sure about something, feel free to ask for details.
