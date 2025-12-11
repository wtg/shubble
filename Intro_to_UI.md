# Introduction to the UI: Beginner Friendly 

This section provides a simple overview of how to access and interact with the project’s user interface. It explains the basic layout, what each part of the UI is responsible for, and how developers can navigate it during testing. Even if you’re new to the project, this guide will help you understand the UI’s role in the overall system.

## Summary

- [Purpose of the UI](#purpose-of-the-ui).
- [How to Access the UI](#how-to-access-the-ui).
- [Frontend Technology Breakdown](#frontend-technology-breakdown).
- [Overview of the UI Layout](#overview-of-the-ui-layout).
- [Key Features Developers Should Know](#key-features-developers-should-know).
- [Developer Workflow with the UI](#developer-workflow-with-the-ui)
- [Troubleshooting Access Issues](#troubleshooting-access-issues)

# UI Overview & Developer 

## Purpose of the UI

The purpose of the UI is to provide a clear, user-friendly way to interact with the project’s main features while also presenting them in a clean and visually appealing layout. It helps developers test functionality, understand how users will experience the system, and ensure that both the interface and the underlying logic work together smoothly.

## How to Access the UI
Verify that all [setup instructions](CONTRIBUTING.md) have been completed first.

To view the UI while you're developing, you’ll start the frontend's development server. Use the following steps below: 
#### 1. Run the command in your terminal:
  In the project’s root folder (the main `shubble` foler) run: 
  ```bash
  npm run dev
  ```
  This command launches a development server so you can run and test your frontend application locally.

#### 2. Wait for the local address to appear:
  Once the dev server starts, your terminal will show a link where the UI is running. It usually looks like:  
  ```bash
  http://localhost:5173/
  ``` 
  The port your development server runs on may differ based on the project’s configuration and whichever ports are free on your system.

  Open the link:
  - **Mac:** `Command + Click` the link the terminal prints.
  - **Windows:** `Ctrl + Click` the link the terminal prints.

  Any changes you make to the frontend code will automatically be reflected in the browser, as the development server supports live reloading.

#### 3. To stop the UI:
Return to the terminal where npm run dev is running and press `Ctrl + C` to terminate the development server.


## Frontend Technology Breakdown
The Shubble UI is built using three main technologies: `TypeScript`, `TSX` (React Components), and `CSS`. Each plays a different role in how the front-end behaves, renders, and looks.

- **TypeScript (`.ts`)**  
  - TypeScript files contain logic that is not directly tied to rendering components -things like utility functions, API calls, data models, type definitions, or shared logic.
  - These files help enforce type safety, reduce bugs, and make the codebase easier to maintain.
  
- **TSX (React Components)(`.tsx`)**  
  - TSX files define the UI itself. TSX is essentially a blend of TypeScript and HTML-like syntax, allowing you to write markup (similar to HTML) directly alongside component logic. This makes it easy to build interactive UI elements such as pages, buttons, forms, and layouts.
  - TSX components control rendering, manage state, handle user events, and communicate with backend services — all within a single, structured file.

- **CSS (`.css`)**  
  - CSS files control styling — layout, colors, spacing, fonts, animations, and responsiveness.
  - They ensure the UI is visually clear, consistent, and user-friendly.


## Overview of the UI Layout
The UI for this project is built using a React + TypeScript front-end located in the `client/` directory. Inside `client/src/`, the layout is divided into logical folders such as `components/` for reusable UI elements, `pages/` for full page views, `styles/` for global and modular CSS, and `data/` or `types/` for shared utilities and type definitions. The main application entry point (`main.tsx`) mounts the UI, while `App.tsx` defines the top-level structure and routing.

From a developer perspective, this layout makes it easy to locate the code responsible for visual components, page-level logic, and shared UI resources. The separation into components and pages also encourages modularity — most UI changes happen inside `client/src/components` or `client/src/pages`, keeping the rest of the system clean and maintainable


## Key Features Developers Should Know


## Developer Workflow with the UI

## Troubleshooting Access Issues
