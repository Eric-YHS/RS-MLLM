import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import 'katex/dist/katex.min.css';
import './Markdown.css';

const cleanSpecialTags = (text) => {
  if (!text) return '';

  const tagsToRemove = [
    '<|begin_of_box|>', '<|end_of_box|>',
    '<|begin_of_text|>', '<|end_of_text|>',
    '<|begin_of_list|>', '<|end_of_list|>',
    '<|begin_of_attribute|>', '<|end_of_attribute|>'
  ];

  let cleanedText = text;
  for (const tag of tagsToRemove) {
    cleanedText = cleanedText.replace(new RegExp(tag, 'g'), '');
  }
  cleanedText = cleanedText.replace(/^\s*\|\|\s*/g, '');
  cleanedText = cleanedText.replace(/\s*\|\|\s*$/g, '');

  cleanedText = cleanedText.replace(/\\\((.*?)\\\)/gs, (match, content) => `$${content}$`);

  cleanedText = cleanedText.replace(/\\\[(.*?)\\\]/gs, (match, content) => `$$${content}$$`);

  return cleanedText;
};

const Markdown = ({ content }) => {
  const cleanedContent = content ? cleanSpecialTags(content) : '';

  return (
    <div className="markdown-content">
      <ReactMarkdown
        remarkPlugins={[remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          code({ node, inline, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || '');
            return !inline && match ? (
              <SyntaxHighlighter
                style={tomorrow}
                language={match[1]}
                PreTag="div"
                {...props}
              >
                {String(children).replace(/\n$/, '')}
              </SyntaxHighlighter>
            ) : (
              <code className={className} {...props}>
                {children}
              </code>
            );
          }
        }}
      >
        {cleanedContent}
      </ReactMarkdown>
    </div>
  );
};

export default Markdown;