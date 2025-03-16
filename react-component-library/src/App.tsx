import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { ImageGrid } from "./components/ImageGrid";
import "./App.css";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { rand, randomNormal } from "@tensorflow/tfjs";

const queryClient = new QueryClient();

function App() {
  const [greetMsg, setGreetMsg] = useState("");
  const [name, setName] = useState("");

  async function greet() {
    // Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
    setGreetMsg(await invoke("greet", { name }));
  }

  return (
    <QueryClientProvider client={queryClient}>
      <ImageGrid
        rows={5}
        cols={10}
        totalRows={1000}
        getImage={async (row, col) => {
          // Simulate an expensive calculation
          await new Promise((resolve) => setTimeout(resolve, 1000));
          if (col % 2 === 1) {
            return `https://picsum.photos/100?random=${row * 10 + col}`;

          } else {
            return `https://picsum.photos/200?random=${row * 10 + col}`;
          }
        }}
        firstVisibleRow={0}
        onScroll={console.log}
      />
    </QueryClientProvider>
  );
}

export default App;
