package io.github.orionlibs.javollama.config;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class MockController
{
    @GetMapping(value = "/", produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<?> getHome(HttpServletRequest request, HttpServletResponse response, Model model)
    {
        return ResponseEntity.ok().body(null);
    }


    @GetMapping(value = "/api/v1/users", produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<?> getUsers(HttpServletRequest request, HttpServletResponse response, Model model)
    {
        return ResponseEntity.ok().body(null);
    }


    @GetMapping(value = "/search", produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<?> search(@RequestParam(name = "query", required = true) String query, @RequestParam(name = "options", required = true) int options, HttpServletRequest request, HttpServletResponse response, Model model)
    {
        return ResponseEntity.ok().body(null);
    }
}