/*
 * Copyright (c) 2014-2026 Bjoern Kimminich & the OWASP Juice Shop contributors.
 * SPDX-License-Identifier: MIT
 */

import { environment } from '../../environments/environment'
import { Injectable, inject } from '@angular/core'
import { HttpClient } from '@angular/common/http'
import { type Observable } from 'rxjs'

export interface AiAssistantMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface AiAssistantResponse {
  reply: string
  blocked: boolean
}

@Injectable({
  providedIn: 'root'
})
export class AiAssistantService {
  private readonly http = inject(HttpClient)
  private readonly host = environment.aiAssistantUrl

  ask (message: string, history: AiAssistantMessage[]): Observable<AiAssistantResponse> {
    return this.http.post<AiAssistantResponse>(`${this.host}/chat`, { message, history })
  }
}
