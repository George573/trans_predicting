import js from "@eslint/js";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import tseslint from "typescript-eslint";
import { defineConfig, globalIgnores } from "eslint/config";
import { projectStructurePlugin } from "eslint-plugin-project-structure";
import { folderStructureConfig } from "./eslint/folder.ts";
import { independentModulesConfig } from "./eslint/modules.ts";

export default defineConfig([
  globalIgnores(["dist"]),
  {
    files: ["**/*.{ts,tsx}"],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      globals: globals.browser,
    },
    plugins: { "project-structure": projectStructurePlugin },
    rules: {
      "project-structure/folder-structure": ["error", folderStructureConfig],
      "project-structure/independent-modules": ["error", independentModulesConfig],
    },
  },
]);
