import React, { useState } from "react";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { motion } from "framer-motion";
import { TextareaAutosize } from "@mui/material";

const SmartNERChatbot = () => {
  const [query, setQuery] = useState("");
  const [text, setText] = useState("");
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleExtract = async () => {
    setLoading(true);
    setResponse(null);

    try {
      const llmResponse = await fetch("/api/llm-handler", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      
      const config = await llmResponse.json(); // Smart model configuration
      
      const nerResponse = await fetch("/api/extract-entities", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, config }),
      });
      
      const result = await nerResponse.json();
      setResponse(result);
    } catch (error) {
      console.error("Error:", error);
    }

    setLoading(false);
  };

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="max-w-2xl mx-auto p-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-xl font-bold text-center">Smart AI NER Chatbot</CardTitle>
        </CardHeader>
        <CardContent>
          <label className="font-semibold">Describe what you need:</label>
          <Textarea
            placeholder="e.g., 'Find all personal data excluding names'"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="mt-2 w-full"
          />
          <label className="font-semibold mt-4 block">Enter text for analysis:</label>
          <TextareaAutosize
            minRows={5}
            placeholder="Paste or type text here..."
            value={text}
            onChange={(e) => setText(e.target.value)}
            className="mt-2 w-full border rounded p-2"
          />
          <Button onClick={handleExtract} disabled={loading} className="mt-4 w-full">
            {loading ? "Processing..." : "Extract Entities"}
          </Button>
          {response && (
            <div className="mt-4 bg-gray-100 p-4 rounded">
              <pre className="whitespace-pre-wrap text-sm">{JSON.stringify(response, null, 2)}</pre>
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
};

export default SmartNERChatbot;
