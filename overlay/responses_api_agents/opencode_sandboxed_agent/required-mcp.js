// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { writeFileSync } from "node:fs";

const defaultSetupErrorPath = "/tmp/nemo-gym-mcp-setup-error";

export default async function requiredMcp({ client }) {
  const required = JSON.parse(process.env.NEMO_GYM_REQUIRED_MCP_SERVERS || "[]");
  let checked = false;
  return {
    "chat.params": async () => {
      if (checked || required.length === 0) return;
      try {
        const { data, error } = await client.mcp.status();
        if (error || required.some((name) => data?.[name]?.status !== "connected")) {
          throw new Error("Required Gym MCP tools could not be initialized");
        }
        checked = true;
      } catch {
        // The host distinguishes setup failure from a scored execution failure.
        writeFileSync(
          process.env.NEMO_GYM_MCP_SETUP_ERROR_PATH || defaultSetupErrorPath,
          "Required Gym MCP tools could not be initialized\n",
        );
        throw new Error("Required Gym MCP tools could not be initialized");
      }
    },
  };
}
