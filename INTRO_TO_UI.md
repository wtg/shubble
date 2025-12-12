# Introduction to the UI: Beginner Friendly 

This section provides a simple overview of how to access and interact with the project’s user interface. It explains the basic layout, what each part of the UI is responsible for, and how developers can navigate it during testing. Even if you’re new to the project, this guide will help you understand the UI’s role in the overall system.

## Summary

- [Purpose of the UI](#purpose-of-the-ui)
- [How the UI Fits Into Shubble](#how-the-ui-fits-into-shubble)
- [How to Access the UI](#how-to-access-the-ui)
- [Frontend Technology Breakdown](#frontend-technology-breakdown)
- [Design and Figma Overview](#design-and-figma-overview)
- [Overview of the UI Layout](#overview-of-the-ui-layout)
- [UI Coding Practices/Paradigms](#ui-coding-practices/paradigms)
- [Key Features Developers Should Know](#key-features-developers-should-know)
- [Troubleshooting Access Issues](#troubleshooting-access-issues)

# UI Overview & Developer Guide

## Purpose of the UI

The purpose of the UI is to provide a clear, user-friendly way to interact with the project’s main features while also presenting them in a clean and visually appealing layout. It helps developers test functionality, understand how users will experience the system, and ensure that both the interface and the underlying logic work together smoothly.

## How the UI Fits Into Shubble

In Shubble, the UI provides a visual and interactive way to view shuttle routes, stops, and system state. It serves as the main point of interaction for understanding how shuttle data is presented to users.

As an example, the UI displays routes and stops on a map, with shuttles shown as moving markers that update over time. Routes are visually distinguished using color, and simple controls such as tabs allow users to switch between routes (e.g., North and West) without reloading the page.

For developers, this makes the UI a quick and intuitive way to test, debug, and validate how changes affect the overall system.

## How to Access the UI
Verify that all [setup instructions](CONTRIBUTING.md) have been completed first.

To view the UI while you're developing, you’ll start the frontend's development server. Use the following steps below: 
#### 1. Run the command in your terminal:
  In the project’s root folder (the main `shubble` folder) run: 
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
  - TypeScript files contain logic that is not directly tied to rendering components - things like utility functions, API calls, data models, type definitions, or shared logic.
  - These files help enforce type safety, reduce bugs, and make the codebase easier to maintain.
  
- **TSX (React Components)(`.tsx`)**  
  - TSX files define the UI itself. TSX is essentially a blend of TypeScript and HTML-like syntax, allowing you to write markup (similar to HTML) directly alongside component logic. This makes it easy to build interactive UI elements such as pages, buttons, forms, and layouts.
  - TSX components control rendering, manage state, handle user events, and communicate with backend services — all within a single, structured file.

- **CSS (`.css`)**  
  - CSS files control styling — layout, colors, spacing, fonts, animations, and responsiveness.
  - They ensure the UI is visually clear, consistent, and user-friendly.

### Helpful Learning Resources 
If you're new to the technologies used in this UI, here are a few resources that will help you understand how the interface is structured and how the components work together:

- https://www.dhiwise.com/post/typescript-essentials-navigating-the-ts-vs-tsx-divide
- https://react.dev/learn/typescript
- https://www.w3schools.com/css/
- https://www.w3schools.com/typescript/typescript_react.php

## Design and Figma Overview
The visual design and layout of the UI are documented in [Figma](https://www.figma.com). These designs serve as a reference for structure, spacing, and visual hierarchy, and help ensure consistency across the interface.

Developers should use the Figma files as a guide when:
- Implementing new UI components
- Modifying existing layouts or styles
- Ensuring visual consistency across pages

The Figma designs are not always a one-to-one mapping with the code, but they provide a strong baseline for how the UI is intended to look and behave.  
To learn how to use figma click [HERE](https://www.youtube.com/watch?v=jQ1sfKIl50E). 
#### Examples of Past Work

<p align="center">
  <img src="https://github.com/user-attachments/assets/6f42795b-9b4d-4fe3-8532-978354d36d8e" width="450"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/020c6f24-a652-4795-8008-79351c54ad63" width="350"/>
</p>



## Overview of the UI Layout
The UI for this project is built using a React + TypeScript front-end located in the `client/` directory. Inside `client/src/`, the layout is divided into logical folders such as `components/` for reusable UI elements, `pages/` for full page views, `styles/` for global and modular CSS, and `data/` or `types/` for shared utilities and type definitions. The main application entry point (`main.tsx`) mounts the UI, while `App.tsx` defines the top-level structure and routing.

From a developer perspective, this layout makes it easy to locate the code responsible for visual components, page-level logic, and shared UI resources. The separation into components and pages also encourages modularity — most UI changes happen inside `client/src/components` or `client/src/pages`, keeping the rest of the system clean and maintainable.

#### Where to Start Editing the UI

If you are new to the UI codebase, a good place to start is:

- `client/src/pages/` — Contains page-level components that represent full views.
- `client/src/components/` — Contains reusable UI components shared across pages.

For small changes or experiments, modifying an existing component in `components/` is usually the safest place to begin. For larger features or layout changes, start with the relevant page in `pages/`.

## UI Coding Practices/Paradigms 

- **CSS-Based Styling (No Inline Styles)**  
  Styling is handled exclusively through CSS files rather than inline styles or style objects within components. This separation keeps presentation concerns isolated from logic, improves readability, and makes global style changes easier to manage.

- **React Functional Components**  
  All UI components are written as React functional components rather than class-based components. This approach aligns with modern React best practices and allows the use of hooks for state management and side effects.

- **Centralized Type Definitions**  
  Major reusable types—especially those related to domain concepts such as Stops or Schedules—are defined in dedicated type files under `client/src/ts/types`. Centralizing these definitions ensures consistency across the UI and reduces duplication.

- **Component Reusability and Modularity**  
  UI logic is broken down into small, reusable components whenever possible. Components are designed to do one thing well and be composed together, rather than embedding large amounts of logic in a single file.

- **Clear Separation of Concerns**  
  Visual presentation, application logic, and data handling are kept as separate as possible. Components focus on rendering and interaction, while shared logic and data structures live in utility or type files.

- **Readable and Explicit Code**  
  Code is written with clarity in mind, favoring explicit variable names and straightforward logic over overly compact or clever solutions. This makes the UI easier to understand and modify, especially for new contributors.

- **Responsiveness and Screen Sizes**  
  When making UI changes, developers should verify that the interface remains usable and visually consistent across different screen widths and window sizes. Components should adapt gracefully to resizing and avoid layouts that break on smaller or larger screens.


Following these practices helps ensure that the UI remains consistent, approachable, and scalable as more features and contributors are added.

## Key Features Developers Should Know

- **Interactive UI Components**  
  The interface includes buttons, inputs, and other interactive elements that allow developers to trigger actions and observe how the system responds. This makes it easier to test functionality without manually invoking backend logic.

- **Live Reloading During Development**  
  When running the UI in development mode, changes to frontend code are automatically reflected in the browser. This enables rapid iteration and immediate visual feedback while developing or debugging UI components.

- **Clear Visual Feedback**  
  The UI provides visible feedback for user actions, such as loading states, updates, or error messages. Developers should use this feedback to confirm that interactions and data flows are working as expected.

- **Component Reusability**  
  The UI is built using reusable components that may be shared across multiple views or pages. Changes to shared components can affect more than one part of the interface, so testing broadly after updates is recommended.

- **Separation of Logic and Presentation**  
  Application logic and data handling are generally separated from visual presentation. This structure allows developers to update functionality or styling independently while keeping the UI maintainable.

## Troubleshooting Access Issues

Below are some common issues developers may encounter when accessing or running the UI, along with suggested solutions.

- **UI does not load in the browser**  
  Ensure the development server is running and that no errors are shown in the terminal. If the server failed to start, try running `npm install` to ensure all dependencies are installed.

- **Command `npm run dev` fails**  
  Check the error message printed in the terminal. Common causes include missing dependencies, an incompatible Node.js version, or syntax errors in the code. Restarting the terminal and re-running the command can sometimes resolve environment issues.

- **Port already in use**  
  If the default port is occupied, the development server will automatically select another available port. Use the URL printed in the terminal rather than assuming a specific port number.

- **Changes not appearing in the browser**  
  Confirm that the development server is still running and that files are being saved. If live reloading stops working, restart the dev server.

- **TypeScript or build errors**  
  Carefully read TypeScript error messages, as they usually point directly to the issue. These errors often stem from incorrect types, missing imports, or outdated interfaces.

- **Blank page or partial UI rendering**  
  Open the browser’s developer console to check for runtime errors. Console messages can help identify missing data, failed network requests, or component rendering issues.

If issues persist, try restarting the development server, clearing the browser cache, or reviewing recent code changes that may have introduced errors.

---
Before making changes, contributors are encouraged to browse existing components and patterns to understand how the UI is structured. Following established conventions helps keep the interface consistent and easier to maintain.

Good luck working with the UI, and happy coding!


