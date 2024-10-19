export const IMAGE_SIZE = {
  width: 96,
  height: 54,
  channels: 1,
} as const;

export const HEADER_BYTE_SIZE = 4;
export const PLAN_INDEX_BYTE_SIZE = 4;
export const IMAGE_BYTE_SIZE =
  IMAGE_SIZE.width * IMAGE_SIZE.height * IMAGE_SIZE.channels;
