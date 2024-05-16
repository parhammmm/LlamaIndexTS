import type {
  ChatMessage,
  MessageContent,
  MessageContentDetail,
} from "../types.js";
import type {
  AnthropicContent,
  AnthropicImageContent,
  AnthropicMediaTypes,
  AnthropicMessage,
  AnthropicTextContent,
} from "./types.js";

const ACCEPTED_IMAGE_MIME_TYPES = [
  "image/jpeg",
  "image/png",
  "image/webp",
  "image/gif",
];

export const mapMessageContentToMessageContentDetails = (
  content: MessageContent,
): MessageContentDetail[] => {
  return Array.isArray(content) ? content : [{ type: "text", text: content }];
};

export const mergeNeighboringSameRoleMessages = (
  messages: AnthropicMessage[],
): AnthropicMessage[] => {
  return messages.reduce(
    (result: AnthropicMessage[], current: AnthropicMessage, index: number) => {
      if (index > 0 && messages[index - 1].role === current.role) {
        result[result.length - 1].content = [
          ...result[result.length - 1].content,
          ...current.content,
        ];
      } else {
        result.push(current);
      }
      return result;
    },
    [],
  );
};

export const mapMessageContentDetailToAnthropicContent = <
  T extends MessageContentDetail,
>(
  detail: T,
): AnthropicContent => {
  let content: AnthropicContent;

  if (detail.type === "text") {
    content = mapTextContent(detail.text);
  } else if (detail.type === "image_url") {
    content = mapImageContent(detail.image_url.url);
  } else {
    throw new Error("Unsupported content detail type");
  }
  return content;
};

export const mapMessageContentToAnthropicContent = <T extends MessageContent>(
  content: T,
): AnthropicContent[] => {
  return mapMessageContentToMessageContentDetails(content).map(
    mapMessageContentDetailToAnthropicContent,
  );
};

export const mapChatMessagesToAnthropicMessages = <T extends ChatMessage>(
  messages: T[],
): AnthropicMessage[] => {
  const mapped = messages
    .flatMap((msg: T): AnthropicMessage[] => {
      return mapMessageContentToMessageContentDetails(msg.content).map(
        (detail: MessageContentDetail): AnthropicMessage => {
          const content = mapMessageContentDetailToAnthropicContent(detail);

          return {
            role: msg.role === "assistant" ? "assistant" : "user",
            content: [content],
          };
        },
      );
    })
    .filter((message: AnthropicMessage) => {
      const content = message.content[0];
      if (content.type === "text" && !content.text) return false;
      if (content.type === "image" && !content.source.data) return false;
      return true;
    });

  return mergeNeighboringSameRoleMessages(mapped);
};

export const mapTextContent = (text: string): AnthropicTextContent => {
  return { type: "text", text };
};

export const extractDataUrlComponents = (
  dataUrl: string,
): {
  mimeType: string;
  base64: string;
} => {
  const parts = dataUrl.split(";base64,");

  if (parts.length !== 2 || !parts[0].startsWith("data:")) {
    throw new Error("Invalid data URL");
  }

  const mimeType = parts[0].slice(5);
  const base64 = parts[1];

  return {
    mimeType,
    base64,
  };
};

export const mapImageContent = (imageUrl: string): AnthropicImageContent => {
  if (!imageUrl.startsWith("data:"))
    throw new Error(
      "For Anthropic please only use base64 data url, e.g.: data:image/jpeg;base64,SGVsbG8sIFdvcmxkIQ==",
    );
  const { mimeType, base64: data } = extractDataUrlComponents(imageUrl);
  if (!ACCEPTED_IMAGE_MIME_TYPES.includes(mimeType))
    throw new Error(
      `Anthropic only accepts the following mimeTypes: ${ACCEPTED_IMAGE_MIME_TYPES.join("\n")}`,
    );

  return {
    type: "image",
    source: {
      type: "base64",
      media_type: mimeType as AnthropicMediaTypes,
      data,
    },
  };
};

export const toUtf8 = (input: Uint8Array): string =>
  new TextDecoder("utf-8").decode(input);
