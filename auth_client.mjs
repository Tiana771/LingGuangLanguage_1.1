let csrfToken = "";
let currentUser = null;
let authMode = "login";
let onAuthenticated = () => {};
let apiBase = "";
let postAuthDestination = "recognize";
let requireLoginAtStart = false;
let avatarDraft = "";

const el = (selector) => document.querySelector(selector);

async function request(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (options.body) headers.set("Content-Type", "application/json");
  if (["POST", "PUT", "PATCH", "DELETE"].includes((options.method || "GET").toUpperCase()) && csrfToken) {
    headers.set("X-CSRF-Token", csrfToken);
  }
  const response = await fetch(`${apiBase}${path}`, { ...options, headers, credentials: "same-origin" });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.message || "账号服务暂不可用，请稍后重试。");
  return payload;
}

function renderUser(user) {
  currentUser = user || null;
  const accountControl = el("#account-control");
  const accountName = el("#account-name");
  accountName.textContent = currentUser ? currentUser.displayName || currentUser.username : "登录 / 注册";
  accountControl.setAttribute("aria-label", currentUser ? "打开个人中心" : "登录或注册账号");
  el("#profile-signed-out").classList.toggle("hidden", Boolean(currentUser));
  el("#profile-content").classList.toggle("hidden", !currentUser);
  renderAvatar(el("#account-avatar"), currentUser?.avatar || "", currentUser?.displayName || currentUser?.username || "");
  document.querySelectorAll("[data-mode='communication'], [data-mode='learning']").forEach((item) => {
    const label = item.dataset.mode === "learning" ? "手语学习" : "沟通助手";
    item.setAttribute("aria-label", currentUser ? label : `${label}，需登录`);
  });
}

function renderAvatar(container, source, label) {
  if (!container) return;
  container.replaceChildren();
  if (source) {
    const image = document.createElement("img");
    image.src = source;
    image.alt = label ? `${label}头像` : "用户头像";
    container.append(image);
    return;
  }
  const initial = document.createElement("span");
  initial.textContent = [...(label || "◉")][0];
  initial.setAttribute("aria-hidden", "true");
  container.append(initial);
}

function renderProfile(profile) {
  currentUser = { ...currentUser, ...profile };
  el("#profile-username").value = profile.username || "";
  el("#profile-display-name").value = profile.displayName || profile.username || "";
  el("#profile-bio").value = profile.bio || "";
  avatarDraft = profile.avatar || "";
  renderAvatar(el("#profile-avatar-preview"), avatarDraft, profile.displayName || profile.username || "");
  renderAvatar(el("#account-avatar"), avatarDraft, profile.displayName || profile.username || "");
  el("#account-name").textContent = profile.displayName || profile.username || "";
}

async function loadProfile() {
  const { profile } = await request("/api/auth/profile");
  renderProfile(profile);
}

async function prepareAvatar(file) {
  if (!file || !["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
    throw new Error("请选择 JPEG、PNG 或 WebP 图片。");
  }
  if (file.size > 10 * 1024 * 1024) throw new Error("原始图片不能超过 10 MB。");
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, 512 / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(bitmap.width * scale));
  canvas.height = Math.max(1, Math.round(bitmap.height * scale));
  canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  let quality = 0.88;
  let encoded = canvas.toDataURL("image/jpeg", quality);
  while (encoded.length > 340_000 && quality > 0.45) {
    quality -= 0.1;
    encoded = canvas.toDataURL("image/jpeg", quality);
  }
  if (encoded.length > 340_000) throw new Error("图片压缩后仍过大，请选择更小的图片。");
  return encoded;
}

function setMode(mode) {
  authMode = mode;
  const register = mode === "register";
  el("#auth-title").textContent = register ? "注册账号" : "登录聆光手语";
  el("#auth-description").textContent = register ? "创建账号后开始使用聆光手语，个人学习和沟通记录会按账号保存。" : "登录后进入手语学习、沟通助手和平台功能。";
  el("#auth-submit").textContent = register ? "注册并登录" : "登录";
  el("#auth-form [name='password']").setAttribute("autocomplete", register ? "new-password" : "current-password");
  document.querySelectorAll("[data-auth-mode]").forEach((button) => button.classList.toggle("active", button.dataset.authMode === mode));
  el("#auth-status").textContent = "";
}

function openDialog(mode = "login") {
  setMode(mode);
  el("#auth-close").classList.toggle("hidden", requireLoginAtStart && !currentUser);
  el("#auth-dialog").classList.remove("hidden");
  el("#auth-form [name='username']").focus();
}

function closeDialog() {
  if (requireLoginAtStart && !currentUser) return;
  el("#auth-dialog").classList.add("hidden");
}

export function getCsrfToken() {
  return csrfToken;
}

export function authenticatedRequest(path, options = {}) {
  return request(path, options);
}

export async function initializeAuth(options = {}) {
  onAuthenticated = options.onAuthenticated || (() => {});
  apiBase = options.apiBase || "";
  requireLoginAtStart = options.requireLoginAtStart === true;
  window.linguangRequest = request;
  try {
    const csrf = await fetch(`${apiBase}/api/auth/csrf`, { credentials: "same-origin" });
    csrfToken = (await csrf.json()).csrfToken || "";
    window.LINGUANG_CSRF_TOKEN = csrfToken;
    const payload = await request("/api/auth/me");
    if (payload.authenticated) {
      renderUser(payload.user);
      try {
        await loadProfile();
      } catch (error) {
        el("#profile-status").textContent = error.message;
      }
      onAuthenticated(payload.user, false);
    } else {
      renderUser(null);
      if (requireLoginAtStart) {
        postAuthDestination = "recognize";
        openDialog("login");
      }
    }
  } catch {
    renderUser(null);
    if (requireLoginAtStart) {
      postAuthDestination = "recognize";
      openDialog("login");
      el("#auth-status").textContent = "暂时无法连接账号服务，请检查服务是否已启动后重试。";
    }
  }

  el("#auth-close").addEventListener("click", closeDialog);
  el("#auth-dialog").addEventListener("click", (event) => {
    if (event.target === el("#auth-dialog") && !(requireLoginAtStart && !currentUser)) closeDialog();
  });
  document.querySelectorAll("[data-auth-mode]").forEach((button) => button.addEventListener("click", () => setMode(button.dataset.authMode)));
  el("#auth-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const status = el("#auth-status");
    const submit = el("#auth-submit");
    submit.disabled = true;
    status.textContent = "正在处理…";
    try {
      const payload = await request(`/api/auth/${authMode === "register" ? "register" : "login"}`, {
        method: "POST",
        body: JSON.stringify({ username: form.get("username"), password: form.get("password") }),
      });
      csrfToken = payload.csrfToken || csrfToken;
      window.LINGUANG_CSRF_TOKEN = csrfToken;
      renderUser(payload.user);
      try {
        await loadProfile();
      } catch (error) {
        el("#profile-status").textContent = error.message;
      }
      closeDialog();
      onAuthenticated(currentUser, true, postAuthDestination);
      postAuthDestination = "recognize";
    } catch (error) {
      status.textContent = error.message;
    } finally {
      submit.disabled = false;
    }
  });
  el("#account-control").addEventListener("click", () => {
    if (!currentUser) {
      postAuthDestination = "profile";
      openDialog("login");
      return;
    }
    window.linguangNavigate?.("profile");
  });
  el("#profile-login").addEventListener("click", () => {
    postAuthDestination = "profile";
    openDialog("login");
  });
  el("#account-logout").addEventListener("click", async () => {
    if (!window.confirm("退出登录？沟通记录仍保存在账号中。")) return;
    try {
      await request("/api/auth/logout", { method: "POST", body: "{}" });
    } finally {
      window.location.reload();
    }
  });
  el("#profile-avatar-choose").addEventListener("click", () => el("#profile-avatar-file").click());
  el("#profile-avatar-file").addEventListener("change", async (event) => {
    const file = event.currentTarget.files?.[0];
    if (!file) return;
    const status = el("#profile-status");
    status.textContent = "正在处理头像…";
    try {
      avatarDraft = await prepareAvatar(file);
      renderAvatar(el("#profile-avatar-preview"), avatarDraft, el("#profile-display-name").value);
      status.textContent = "头像已准备，点击保存资料后生效。";
    } catch (error) {
      status.textContent = error.message;
    } finally {
      event.currentTarget.value = "";
    }
  });
  el("#profile-avatar-remove").addEventListener("click", () => {
    avatarDraft = "";
    renderAvatar(el("#profile-avatar-preview"), "", el("#profile-display-name").value);
    el("#profile-status").textContent = "头像已移除，点击保存资料后生效。";
  });
  el("#profile-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const status = el("#profile-status");
    const submit = event.currentTarget.querySelector("button[type='submit']");
    submit.disabled = true;
    status.textContent = "正在保存…";
    try {
      const { profile } = await request("/api/auth/profile", {
        method: "PUT",
        body: JSON.stringify({
          displayName: el("#profile-display-name").value,
          bio: el("#profile-bio").value,
          avatar: avatarDraft,
        }),
      });
      renderProfile(profile);
      status.textContent = "资料已保存。";
    } catch (error) {
      status.textContent = error.message;
    } finally {
      submit.disabled = false;
    }
  });
  el("#password-change-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const status = el("#password-change-status");
    try {
      const payload = await request("/api/auth/password", {
        method: "POST",
        body: JSON.stringify({ currentPassword: form.get("currentPassword"), newPassword: form.get("newPassword") }),
      });
      status.textContent = payload.message || "密码已更新。";
      event.currentTarget.reset();
    } catch (error) {
      status.textContent = error.message;
    }
  });
  window.linguangAuth = {
    currentUser: () => currentUser,
    requireLogin: (destination = "communication") => {
      if (currentUser) return true;
      postAuthDestination = destination;
      openDialog("login");
      return false;
    },
  };
}
