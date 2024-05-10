FROM node:20-alpine as build-stage

WORKDIR /app

COPY package*.json .

RUN npm install

COPY . .

RUN npm run build

# Multi-stage build
FROM nginx:1.25 as production-stage

RUN mkdir /app 

COPY --from=build-stage /app/build /app

COPY nginx.conf /etc/nginx/nginx.conf