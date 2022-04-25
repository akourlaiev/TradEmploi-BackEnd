import axios from "axios"

import { userInfosResponse } from "../models/userInfosResponse";
require('dotenv').config()
// process.env['NODE_TLS_REJECT_UNAUTHORIZED'] = '0' 

export const getUserInfo = async (accessToken:string) => {
  
  const config = {
    headers: {
      "Authorization": "Bearer "+accessToken,
    },
  };
  return axios
    .post(process.env.USER_INFO_URL ?? "", null, config)
    .then((response) => {
      return response.data as unknown as userInfosResponse;
    })
    .catch(function (error) {
      console.error(error);
    });
}